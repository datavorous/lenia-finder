#!/usr/bin/env python3
"""Render verified Lenia configurations on a true, scaled 1024 grid.

The 128-grid verification snapshot at step 5,000 is repeated by the scale
factor to initialize the larger grid. Subsequent frames are actual Lenia
updates with the original rule, timestep, and kernel scaled through the
simulator's existing world_scale setting. This is a scale-transfer experiment;
its trajectory is not an exact continuation of the 128-grid discretization.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy
from PIL import Image

from simulation.lenia import Lenia


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("archive/runs/soliton-work/soliton-results/catalog.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/soliton-work/moving-highres-local")
    )
    parser.add_argument("--grid", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument(
        "--include-oscillators",
        action="store_true",
        help="Render all robust verified configurations",
    )
    args = parser.parse_args()
    if args.grid % 128 or args.frames % args.chunk_frames:
        parser.error("grid must be a multiple of 128; frames must divide into chunks")
    if jax.default_backend() != "cpu":
        parser.error("This local renderer requires the CPU JAX backend")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.catalog.open(newline="") as handle:
        candidates = [
            r
            for r in csv.DictReader(handle)
            if r["robust_verified"] == "True"
            and (args.include_oscillators or r["category"] == "moving_soliton")
        ]
    if not candidates:
        parser.error("No matching robust configurations in catalog")

    scale = args.grid // 128
    substrate = Lenia(
        grid_size=args.grid,
        phenotype_size=128,
        start_pattern="5N7KKM",
        clip1=1.0,
        world_scale=scale,
    )
    radius = substrate.lenia.pattern["R"] * scale

    def step(state, params):
        return substrate.step_state(None, state, params)

    @jax.jit
    def render_chunk(state, params):
        def frame(current, _):
            def one(s, unused):
                return step(s, params), None

            current, _ = jax.lax.scan(one, current, None, length=args.steps_per_frame)
            world = current["carry"].world
            center = (
                current["carry"].temp.last_center * radius
                + current["carry"].temp.total_shift
            )
            mass = jnp.mean(world)
            occupied = jnp.mean(jnp.max(world, axis=-1) > 0.05)
            return current, (world, center, mass, occupied)

        return jax.lax.scan(frame, state, None, length=args.chunk_frames)

    cards = []
    for row in candidates:
        kind, ident = row["source_kind"], int(row["source_id"])
        stem = f"{kind}-{ident:08d}-{args.grid}"
        video = args.output / f"{stem}.mp4"
        metadata = args.output / f"{stem}.json"
        poster = args.output / f"{stem}.png"
        params_path = (args.catalog.parent / row["parameter_file"]).resolve()
        trials = (
            args.catalog.parent
            / row["verification_folder"]
            / "control_trajectories.npz"
        ).resolve()
        if video.exists() and metadata.exists() and poster.exists():
            print(f"already rendered {kind} {ident}", flush=True)
            cards.append(
                dict(
                    source_kind=kind,
                    source_id=ident,
                    category=row["category"],
                    video=video.name,
                    poster=poster.name,
                    metadata=metadata.name,
                )
            )
            continue
        params = np.load(params_path).astype(np.float32)
        with np.load(trials) as saved:
            # Snapshots are captured at steps 1,000, 2,000, ..., 10,000.
            start_128 = saved["snapshots"][0, 4].astype(np.float32)
        np.save(args.output / f"{stem}-parameters.npy", params)
        np.save(args.output / f"{stem}-start-step5000-128.npy", start_128)
        world = jnp.repeat(jnp.repeat(jnp.asarray(start_128), scale, 0), scale, 1)
        key = jax.random.fold_in(jax.random.PRNGKey(424242), ident)
        state = substrate.init_state(key, jnp.asarray(params))
        temp = state["carry"].temp
        zero_temp = type(temp)(*(jnp.zeros_like(v) for v in temp))
        state = dict(
            carry=state["carry"]._replace(world=world, temp=zero_temp), img=state["img"]
        )

        temp_video = args.output / f"{stem}.tmp.mp4"
        measurements = []
        final_rgb = None
        with imageio.get_writer(
            temp_video, fps=args.fps, codec="libx264", quality=8, macro_block_size=16
        ) as writer:
            for chunk in range(args.frames // args.chunk_frames):
                state, (worlds, centers, masses, areas) = render_chunk(
                    state, jnp.asarray(params)
                )
                worlds, centers, masses, areas = jax.device_get(
                    (worlds, centers, masses, areas)
                )
                for i in range(args.chunk_frames):
                    frame_index = chunk * args.chunk_frames + i
                    rgb = np.uint8(np.clip(worlds[i] * 255.0, 0, 255))
                    writer.append_data(rgb)
                    final_rgb = rgb
                    measurements.append(
                        dict(
                            frame=frame_index + 1,
                            simulation_step=5000
                            + (frame_index + 1) * args.steps_per_frame,
                            center_y_px=float(centers[i, 0]),
                            center_x_px=float(centers[i, 1]),
                            mass_fraction=float(masses[i]),
                            occupied_fraction=float(areas[i]),
                        )
                    )
                print(
                    f"{kind} {ident}: {len(measurements)}/{args.frames} frames",
                    flush=True,
                )
        temp_video.replace(video)
        Image.fromarray(final_rgb).save(poster)
        np.save(
            args.output / f"{stem}-final-world.npy",
            np.asarray(jax.device_get(state["carry"].world)),
        )
        save_csv(args.output / f"{stem}-metrics.csv", measurements)
        centers = np.asarray(
            [[r["center_y_px"], r["center_x_px"]] for r in measurements]
        )
        masses = np.asarray([r["mass_fraction"] for r in measurements])
        area = np.asarray([r["occupied_fraction"] for r in measurements])
        info = dict(
            source_kind=kind,
            source_id=ident,
            native_class=row["category"],
            source_rank=int(row["rank"]),
            source_parameter_file=str(params_path),
            source_parameter_sha256=digest(params_path),
            source_control_file=str(trials),
            source_control_sha256=digest(trials),
            simulator_wrapper_sha256=digest(
                Path(__file__).resolve().parents[1] / "simulation/lenia.py"
            ),
            simulator_core_sha256=digest(
                Path(__file__).resolve().parents[1] / "simulation/dynamics.py"
            ),
            renderer_sha256=digest(Path(__file__).resolve()),
            initial_snapshot_step=5000,
            initial_snapshot_index=4,
            initial_snapshot_grid=128,
            initial_snapshot_upscale="nearest pixel repeat",
            rng_base_seed=424242,
            rng_fold_in_source_id=ident,
            carry_motion_state_reset_to_zero=True,
            highres_grid=args.grid,
            world_scale=scale,
            kernel_radius_px=radius,
            simulation_steps=args.frames * args.steps_per_frame,
            frames=args.frames,
            steps_per_frame=args.steps_per_frame,
            fps=args.fps,
            duration_seconds=args.frames / args.fps,
            native_lenia_timestep_unchanged=True,
            jax_version=jax.__version__,
            jaxlib_version=jaxlib.__version__,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            backend=jax.default_backend(),
            mass_min=float(masses.min()),
            mass_max=float(masses.max()),
            occupied_min=float(area.min()),
            occupied_max=float(area.max()),
            net_displacement_px=float(np.linalg.norm(centers[-1] - centers[0])),
            net_displacement_native_px=float(
                np.linalg.norm(centers[-1] - centers[0]) / scale
            ),
            note="True high-resolution scale-transfer simulation from a repeated native-grid checkpoint; trajectory need not equal the native-grid continuation",
        )
        metadata.write_text(json.dumps(info, indent=2) + "\n")
        cards.append(
            dict(
                source_kind=kind,
                source_id=ident,
                category=row["category"],
                video=video.name,
                poster=poster.name,
                metadata=metadata.name,
            )
        )

    html = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        "<title>High-resolution verified Lenia</title>",
        "<style>body{background:#111;color:#eee;font:16px system-ui;margin:24px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}"
        "video{width:100%}a{color:#9bd}</style>",
        "<h1>Verified Lenia at true high resolution</h1>",
        "<p>Each 10-second video contains 1,000 actual Lenia updates on a "
        f"{args.grid}×{args.grid} grid, initialized from a saved 5,000-step "
        "native-grid snapshot. The scaled trajectory can differ from the "
        "native-grid continuation.</p><main>",
    ]
    for card in cards:
        label = (
            "multi-component aggregate"
            if card["source_kind"] == "QD" and card["source_id"] == 547444
            else card["category"].replace("_", " ")
        )
        html.append(
            f"<article><h2>{card['source_kind']} {card['source_id']}</h2>"
            f"<p>Native classification: {label}</p>"
            f'<video controls preload="metadata" poster="{card["poster"]}" '
            f'src="{card["video"]}"></video><p><a href="{card["metadata"]}">'
            "Parameters and measurements</a></p></article>"
        )
    html.append("</main></html>")
    (args.output / "index.html").write_text("\n".join(html))
    (args.output / "gallery.json").write_text(json.dumps(cards, indent=2) + "\n")


if __name__ == "__main__":
    main()
