"""Supervised Lenia discovery using ASAL's Sep-CMA-ES optimization."""

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--device=cpu" in sys.argv or any(
    sys.argv[i] == "--device" and i + 1 < len(sys.argv) and sys.argv[i + 1] == "cpu"
    for i in range(len(sys.argv))
):
    os.environ["JAX_PLATFORMS"] = "cpu"

import evosax
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image

from evaluation.embeddings import MODELS, Encoder, target_loss
from simulation.lenia import Lenia
from simulation.trajectory import sampled_frames


def arguments():
    parser = argparse.ArgumentParser(
        description="Search Lenia configurations for a text target"
    )
    parser.add_argument(
        "prompt", help="Target text; separate temporal targets with semicolons"
    )
    parser.add_argument("--encoder", choices=MODELS, default="mobileclip2-s0")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--simulation-batch", type=int, default=4)
    parser.add_argument("--inference-batch", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs/lenia"))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def validate(args):
    prompts = [part.strip() for part in args.prompt.split(";")]
    if not all(prompts):
        raise ValueError("Each prompt must contain text")
    for key in (
        "iterations",
        "population",
        "simulation_batch",
        "inference_batch",
        "rollout_steps",
        "samples",
        "checkpoint_every",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.samples < len(prompts) or args.samples % len(prompts):
        raise ValueError("samples must be a multiple of the number of prompts")
    if args.rollout_steps % args.samples:
        raise ValueError("rollout-steps must be divisible by samples")
    if args.sigma <= 0 or (args.max_minutes is not None and args.max_minutes <= 0):
        raise ValueError("sigma and max-minutes must be positive")
    return prompts


def atomic_pickle(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def torch_frames(jax_frames):
    # DLPack shares device memory when JAX and Torch use the same CUDA device.
    return torch.from_dlpack(jax_frames)


def save_best(output, params, fitness, frames, video_frames, history, config):
    np.savez_compressed(
        output / "best.npz",
        params=np.asarray(params),
        loss=np.asarray(fitness),
        frames=np.asarray(frames),
    )
    (output / "scores.json").write_text(json.dumps(history, indent=2))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    images = [
        Image.fromarray(np.uint8(np.clip(frame, 0, 1) * 255)).resize(
            (256, 256), Image.Resampling.NEAREST
        )
        for frame in np.asarray(frames)
    ]
    images[-1].save(output / "preview.png")
    if len(images) > 1:
        images[0].save(
            output / "trajectory.gif",
            save_all=True,
            append_images=images[1:],
            duration=150,
            loop=0,
        )
    video = [np.uint8(np.clip(frame, 0, 1) * 255) for frame in np.asarray(video_frames)]
    imageio.mimsave(
        output / "trajectory.mp4", video, fps=16, macro_block_size=16, quality=8
    )


def run(args):
    prompts = validate(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; specify --device cpu only for a CPU run"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        prompt=args.prompt,
        encoder=args.encoder,
        iterations=args.iterations,
        population=args.population,
        simulation_batch=args.simulation_batch,
        inference_batch=args.inference_batch,
        rollout_steps=args.rollout_steps,
        samples=args.samples,
        seed=args.seed,
        sigma=args.sigma,
        device=args.device,
    )
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"{checkpoint} exists; use --resume or a new --output")
    encoder = Encoder(args.encoder, args.device, not args.no_mixed_precision)
    config.update(
        python=platform.python_version(),
        jax=jax.__version__,
        torch=torch.__version__,
        model_revision=encoder.model_revision,
        checkpoint_sha256=encoder.checkpoint_sha256,
        gpu_name=(torch.cuda.get_device_name(0) if args.device == "cuda:0" else None),
    )
    target = encoder.text(prompts)
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    dimensions = int(substrate.default_params(jax.random.PRNGKey(0)).size)
    strategy = evosax.Sep_CMA_ES(
        popsize=args.population, num_dims=dimensions, sigma_init=args.sigma
    )
    es_params = strategy.default_params
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = strategy.initialize(init_rng, es_params)
    start_iteration, history = 0, []
    if args.resume:
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        old_config = saved["config"]
        for key in (
            "prompt",
            "encoder",
            "population",
            "rollout_steps",
            "samples",
            "seed",
            "sigma",
        ):
            if config[key] != old_config[key]:
                raise ValueError(f"Cannot change {key} when resuming")
        state, rng = saved["state"], saved["rng"]
        start_iteration, history = saved["iteration"], saved["history"]

    @jax.jit
    def ask(es_state, key):
        return strategy.ask(key, es_state, es_params)

    @jax.jit
    def tell(candidates, losses, es_state):
        return strategy.tell(candidates, losses, es_state, es_params)

    @jax.jit
    def simulate(candidates, key):
        return jax.vmap(
            lambda p: sampled_frames(
                substrate, key, p, args.rollout_steps, args.samples
            )
        )(candidates)

    @jax.jit
    def full_video(params, key):
        return sampled_frames(
            substrate, key, params, args.rollout_steps, args.rollout_steps
        )

    began = time.monotonic()
    for iteration in range(start_iteration, args.iterations):
        iteration_began = time.monotonic()
        if args.max_minutes is not None and iteration > start_iteration:
            if time.monotonic() - began >= args.max_minutes * 60:
                break
        rng, ask_rng, sim_rng = jax.random.split(rng, 3)
        candidates, next_state = ask(state, ask_rng)
        scores = []
        for offset in range(0, args.population, args.simulation_batch):
            batch = candidates[offset : offset + args.simulation_batch]
            frames = simulate(batch, sim_rng)
            embedding = encoder.images(
                torch_frames(frames.reshape((-1, 64, 64, 3))), args.inference_batch
            )
            embedding = embedding.reshape((len(batch), args.samples, -1))
            loss = target_loss(embedding, target)
            scores.append(loss.cpu().numpy())
        losses = jnp.asarray(np.concatenate(scores), dtype=jnp.float32)
        state = tell(candidates, losses, next_state)
        best_loss = float(state.best_fitness)
        history.append(
            dict(
                iteration=iteration + 1,
                best_loss=best_loss,
                population_mean_loss=float(np.mean(np.concatenate(scores))),
                iteration_seconds=round(time.monotonic() - iteration_began, 3),
            )
        )
        print(
            f"{iteration + 1}/{args.iterations} best loss {best_loss:.5f} "
            f"seconds {history[-1]['iteration_seconds']:.1f}",
            flush=True,
        )
        if (iteration + 1) % args.checkpoint_every == 0:
            atomic_pickle(
                checkpoint,
                dict(
                    state=state,
                    rng=rng,
                    iteration=iteration + 1,
                    history=history,
                    config=config,
                ),
            )

    atomic_pickle(
        checkpoint,
        dict(
            state=state, rng=rng, iteration=len(history), history=history, config=config
        ),
    )
    best = np.asarray(state.best_member)
    best_frames = simulate(jnp.asarray(best)[None], rng)[0]
    video_frames = full_video(jnp.asarray(best), rng)
    save_best(
        output,
        best,
        float(state.best_fitness),
        np.asarray(best_frames),
        np.asarray(video_frames),
        history,
        config,
    )
    print(f"Saved results to {output}")


if __name__ == "__main__":
    run(arguments())
