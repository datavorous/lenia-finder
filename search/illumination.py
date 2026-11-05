"""ASAL Lenia illumination with MobileCLIP2-S2 embeddings and a large archive."""

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
import torch
from PIL import Image

from evaluation.embeddings import Encoder
from simulation.lenia import Lenia
from simulation.trajectory import final_frame, sampled_frames, sampled_frames_with_final


def parse_args():
    parser = argparse.ArgumentParser(description="ASAL illumination on Lenia")
    parser.add_argument("--output", type=Path, default=Path("runs/illumination-s2"))
    parser.add_argument("--archive-size", type=int, default=5000)
    parser.add_argument("--children", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--simulation-batch", type=int, default=32)
    parser.add_argument("--inference-batch", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--trajectory-batch", type=int, default=32)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--representative-videos", type=int, default=24)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-trajectories", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    for key in (
        "archive_size",
        "children",
        "iterations",
        "rollout_steps",
        "simulation_batch",
        "inference_batch",
        "checkpoint_every",
        "trajectory_batch",
        "samples",
    ):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.sigma <= 0 or args.children >= args.archive_size:
        parser.error(
            "sigma must be positive and children must be smaller than archive size"
        )
    if args.rollout_steps % args.samples:
        parser.error("rollout steps must be divisible by samples")
    if args.representative_videos < 0 or args.representative_videos > args.archive_size:
        parser.error("invalid representative video count")
    return args


def atomic_pickle(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def prepare_functions(substrate, args, dimensions):
    population = args.archive_size
    children = args.children

    @jax.jit
    def propose(params, rng):
        rng, parent_rng = jax.random.split(rng)
        parents = jax.random.randint(parent_rng, (children,), 0, population)
        rng, noise_rng, _ = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, (children, dimensions))
        child_params = params[parents] + args.sigma * noise
        rng, rollout_rng = jax.random.split(rng)
        keys = jax.random.split(rollout_rng, children)
        return child_params, keys

    @jax.jit
    def simulate(params, keys):
        return jax.vmap(
            lambda p, key: final_frame(substrate, key, p, args.rollout_steps)
        )(params, keys)

    @jax.jit
    def trace(params, keys):
        return jax.vmap(
            lambda p, key: sampled_frames_with_final(
                substrate, key, p, args.rollout_steps, args.samples
            )
        )(params, keys)

    @jax.jit
    def video(params, key):
        return sampled_frames(
            substrate, key, params, args.rollout_steps, args.rollout_steps
        )

    return propose, simulate, trace, video


@torch.inference_mode()
def select_archive(pop_params, pop_z, child_params, child_z, children):
    """ASAL's iterative two-neighbor cull, with one GPU top-k pass per pool."""
    pool_z = torch.cat((pop_z, child_z), dim=0)
    cosine = pool_z @ pool_z.T
    cosine.fill_diagonal_(-torch.inf)
    # At most `children` neighbors can be removed. Therefore the two closest
    # survivors must be within the initial children+2 nearest neighbors.
    nearest_values, nearest_indices = cosine.topk(children + 2, dim=1)
    alive = torch.ones(len(pool_z), device=pool_z.device, dtype=torch.bool)
    for _ in range(children):
        values = nearest_values.masked_fill(~alive[nearest_indices], -torch.inf)
        two_nearest = values.topk(2, dim=1).values.mean(dim=1)
        kill = two_nearest.masked_fill(~alive, -torch.inf).argmax()
        alive[kill] = False
    kept = torch.nonzero(alive).flatten()
    nearest = (
        nearest_values.masked_fill(~alive[nearest_indices], -torch.inf)
        .max(dim=1)
        .values[kept]
    )
    kept_jax = jax.dlpack.from_dlpack(kept.to(torch.int32).contiguous())
    pool_params = jnp.concatenate((pop_params, child_params), axis=0)
    return pool_params[kept_jax], pool_z[kept], nearest.mean()


@torch.inference_mode()
def final_neighbors(z):
    cosine = z @ z.T
    cosine.fill_diagonal_(-torch.inf)
    return cosine.topk(2, dim=1).values


def checkpoint_state(path, params, z, rng, iteration, history, config):
    atomic_pickle(
        path,
        dict(
            params=np.asarray(jax.device_get(params)),
            embeddings=z.detach().cpu().numpy(),
            rng=np.asarray(jax.device_get(rng)),
            iteration=iteration,
            history=history,
            config=config,
        ),
    )


def representatives(z, novelty, count):
    if not count:
        return []
    z = np.asarray(z)
    chosen = [int(np.argmax(novelty))]
    nearest = z @ z[chosen[0]]
    for _ in range(1, count):
        candidate = int(np.argmin(nearest))
        chosen.append(candidate)
        nearest = np.maximum(nearest, z @ z[candidate])
    return chosen


def save_trajectories(output, params, rng, trace, video, args, z, novelty):
    signature = hashlib.sha256(params.tobytes()).hexdigest()
    signature_path = output / "trajectory_archive.sha256"
    if not signature_path.exists() or signature_path.read_text().strip() != signature:
        for stale in (output / "trajectories", output / "representative_videos"):
            if stale.exists():
                shutil.rmtree(stale)
        (output / "finalized").unlink(missing_ok=True)
        signature_path.write_text(signature + "\n")
    trajectory_dir = output / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    selected = representatives(z, novelty, args.representative_videos)
    (output / "representatives.json").write_text(json.dumps(selected, indent=2))
    chosen_frames = {}

    for start in range(0, args.archive_size, args.trajectory_batch):
        stop = min(start + args.trajectory_batch, args.archive_size)
        path = trajectory_dir / f"frames-{start:05d}-{stop:05d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                final_images = saved["final"]
        else:
            batch = jnp.asarray(params[start:stop])
            keys = jax.random.split(jax.random.fold_in(rng, start), len(batch))
            frames, final = trace(batch, keys)
            frames = np.uint8(np.clip(np.asarray(jax.device_get(frames)), 0, 1) * 255)
            final_images = np.uint8(
                np.clip(np.asarray(jax.device_get(final)), 0, 1) * 255
            )
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp,
                sampled=frames,
                final=final_images,
                indices=np.arange(start, stop, dtype=np.int32),
            )
            temp.replace(path)
        for index in selected:
            if start <= index < stop:
                chosen_frames[index] = final_images[index - start]
        print(f"Trajectories {stop}/{args.archive_size}", flush=True)

    video_dir = output / "representative_videos"
    video_dir.mkdir(exist_ok=True)
    for number, index in enumerate(selected, start=1):
        path = video_dir / f"{number:02d}-archive-{index:04d}.mp4"
        if not path.exists():
            frames = np.asarray(
                jax.device_get(
                    video(jnp.asarray(params[index]), jax.random.fold_in(rng, index))
                )
            )
            frames = np.uint8(np.clip(frames, 0, 1) * 255)
            imageio.mimsave(path, frames, fps=16, macro_block_size=16, quality=8)
        preview = path.with_suffix(".png")
        if not preview.exists():
            Image.fromarray(chosen_frames[index]).resize(
                (256, 256), Image.Resampling.NEAREST
            ).save(preview)
        print(f"Representative video {number}/{len(selected)}", flush=True)
    (output / "finalized").touch()


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"{checkpoint} exists; use --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for illumination")
    encoder = Encoder("mobileclip2-s2", "cuda:0")
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    dimensions = int(substrate.default_params(jax.random.PRNGKey(0)).size)
    propose, simulate, trace, video = prepare_functions(substrate, args, dimensions)
    config = dict(
        archive_size=args.archive_size,
        children=args.children,
        iterations=args.iterations,
        sigma=args.sigma,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        simulation_batch=args.simulation_batch,
        inference_batch=args.inference_batch,
        checkpoint_every=args.checkpoint_every,
        trajectory_batch=args.trajectory_batch,
        samples=args.samples,
        representative_videos=args.representative_videos,
        encoder="apple/MobileCLIP2-S2",
        model_revision=encoder.model_revision,
        checkpoint_sha256=encoder.checkpoint_sha256,
        source_asal_commit="677ba0ea4d3b3ca78273c9906c6e84d2b1481ce7",
        python=platform.python_version(),
        jax=jax.__version__,
        torch=str(torch.__version__),
        gpu_name=torch.cuda.get_device_name(0),
        selection="ASAL 2-neighbor iterative least-novel removal",
        objective="mean nearest-neighbor cosine (minimize)",
        init="5000 zero parameter vectors; one identical Lenia rollout reused",
    )
    if args.resume:
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        for key in (
            "archive_size",
            "children",
            "sigma",
            "seed",
            "rollout_steps",
            "encoder",
        ):
            if saved["config"][key] != config[key]:
                raise ValueError(f"Cannot change {key} on resume")
        (output / "config.json").write_text(json.dumps(config, indent=2))
        params = jax.device_put(saved["params"])
        z = torch.from_numpy(saved["embeddings"]).to("cuda:0")
        rng = jax.device_put(saved["rng"])
        start_iteration = saved["iteration"]
        history = saved["history"]
    else:
        (output / "config.json").write_text(json.dumps(config, indent=2))
        rng = jax.random.PRNGKey(args.seed)
        rng, initial_rng = jax.random.split(rng)
        zero = jnp.zeros((1, dimensions), dtype=jnp.float32)
        frame = simulate(zero, initial_rng[None])
        embedding = encoder.images(torch.from_dlpack(frame), 1)
        params = jnp.broadcast_to(zero, (args.archive_size, dimensions))
        z = embedding.expand(args.archive_size, -1).contiguous()
        start_iteration, history = 0, []
        checkpoint_state(checkpoint, params, z, rng, 0, history, config)

    began = time.monotonic()
    for iteration in range(start_iteration, args.iterations):
        if (
            args.max_minutes
            and iteration > start_iteration
            and time.monotonic() - began >= args.max_minutes * 60
        ):
            break
        iteration_start = time.monotonic()
        rng, iteration_rng = jax.random.split(rng)
        child_params, keys = propose(params, iteration_rng)
        if args.profile:
            child_params.block_until_ready()
            proposal_seconds = time.monotonic() - iteration_start
        embeddings = []
        for offset in range(0, args.children, args.simulation_batch):
            frames = simulate(
                child_params[offset : offset + args.simulation_batch],
                keys[offset : offset + args.simulation_batch],
            )
            features = encoder.images(torch.from_dlpack(frames), args.inference_batch)
            embeddings.append(features)
        child_z = torch.cat(embeddings, dim=0)
        if args.profile:
            torch.cuda.synchronize()
            simulation_seconds = time.monotonic() - iteration_start - proposal_seconds
        params, z, loss = select_archive(
            params, z, child_params, child_z, args.children
        )
        record = dict(
            iteration=iteration + 1,
            illumination_loss=float(loss.item()),
            iteration_seconds=round(time.monotonic() - iteration_start, 3),
        )
        if args.profile:
            record.update(
                proposal_seconds=round(proposal_seconds, 3),
                simulation_seconds=round(simulation_seconds, 3),
                selection_seconds=round(
                    record["iteration_seconds"] - proposal_seconds - simulation_seconds,
                    3,
                ),
            )
        history.append(record)
        if (iteration + 1) % args.checkpoint_every == 0:
            checkpoint_state(checkpoint, params, z, rng, iteration + 1, history, config)
        if (iteration + 1) % 10 == 0 or iteration == start_iteration:
            print(
                f"{iteration + 1}/{args.iterations} loss {record['illumination_loss']:.6f} "
                f"seconds {record['iteration_seconds']:.2f}",
                flush=True,
            )

    completed = len(history)
    checkpoint_state(checkpoint, params, z, rng, completed, history, config)
    (output / "history.json").write_text(json.dumps(history, indent=2))
    params_np = np.asarray(jax.device_get(params))
    z_np = z.detach().cpu().numpy()
    np.save(output / "parameters.npy", params_np)
    np.save(output / "embeddings.npy", z_np)
    nearest = final_neighbors(z).detach().cpu().numpy()
    np.save(output / "nearest_cosine.npy", nearest)
    novelty = 1.0 - nearest[:, 0]
    np.save(output / "novelty.npy", novelty)
    print(
        f"Saved archive: {len(params_np)} configurations, {completed} iterations, "
        f"illumination loss {nearest[:, 0].mean():.6f}",
        flush=True,
    )
    if completed == args.iterations and not args.skip_trajectories:
        save_trajectories(output, params_np, rng, trace, video, args, z_np, novelty)


if __name__ == "__main__":
    main()
