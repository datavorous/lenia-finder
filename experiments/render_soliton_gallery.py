#!/usr/bin/env python3
"""Render ranked 128-grid Lenia candidates as 10-second videos."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from simulation.lenia import Lenia


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalog", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--display-size", type=int, default=1024)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    if jax.default_backend() != "gpu":
        p.error("GPU backend required")
    if args.display_size % 128:
        p.error("display-size must be a multiple of 128")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.catalog.open(newline="") as f:
        rows = list(csv.DictReader(f))[: args.limit]
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=128,
        start_pattern="5N7KKM",
        clip1=1.0,
    )

    @jax.jit
    def simulate(params, key):
        state = substrate.init_state(key, params)

        def one_step(s, _):
            return substrate.step_state(None, s, params), None

        state, _ = jax.lax.scan(one_step, state, None, length=4999)

        def frame(s, _):
            s, _ = jax.lax.scan(one_step, s, None, length=5)
            return s, s["carry"].world

        _, frames = jax.lax.scan(frame, state, None, length=200)
        return frames

    cards = []
    for position, r in enumerate(rows, 1):
        source_id = int(r["source_id"])
        source_kind = r["source_kind"]
        stem = f"{position:03d}-{source_kind}-{source_id:08d}"
        mp4 = args.output / f"{stem}.mp4"
        png = args.output / f"{stem}.png"
        if not mp4.exists() or not png.exists():
            key = jax.random.fold_in(jax.random.PRNGKey(424242), source_id)
            parameter_path = Path(r["parameter_file"])
            if not parameter_path.is_absolute():
                parameter_path = args.catalog.parent / parameter_path
            frames = np.asarray(
                jax.device_get(simulate(jnp.asarray(np.load(parameter_path)), key))
            )
            temp = args.output / f"{stem}.tmp.mp4"
            with imageio.get_writer(
                temp, fps=args.fps, codec="libx264", quality=8, macro_block_size=16
            ) as writer:
                for frame in frames:
                    rgb = Image.fromarray(np.uint8(np.clip(frame, 0, 1) * 255))
                    rgb = rgb.resize(
                        (args.display_size, args.display_size), Image.Resampling.NEAREST
                    )
                    writer.append_data(np.asarray(rgb))
            temp.replace(mp4)
            rgb.save(png)
            print(
                f"video {position}/{len(rows)}: {source_kind} {source_id}", flush=True
            )
        cards.append(
            dict(
                rank=position,
                source_kind=source_kind,
                source_id=source_id,
                category=r["category"],
                speed=float(r["last_1000_speed_px_per_step"]),
                robust=r["robust_verified"],
                damage=float(r["damage_survival_fraction"]),
                noise=float(r["noise_survival_fraction"]),
                video=mp4.name,
                poster=png.name,
            )
        )
    config = dict(
        source_catalog=str(args.catalog.resolve()),
        source_archive=str(args.archive.resolve()),
        count=len(cards),
        native_grid=128,
        display_size=args.display_size,
        warmup_steps=5000,
        recorded_steps=1000,
        frames=200,
        frame_period_steps=5,
        fps=args.fps,
        display_method="nearest-neighbor enlargement; dynamics stay on native 128 grid",
    )
    (args.output / "gallery_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    (args.output / "gallery.json").write_text(json.dumps(cards, indent=2) + "\n")
    html = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Ranked Lenia candidates</title>",
        "<style>body{background:#101118;color:#eee;font:16px system-ui;margin:24px}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}"
        "article{background:#1c1e29;padding:12px}video{width:100%;aspect-ratio:1}"
        "a{color:#9bd}</style>",
        "<h1>Ranked Lenia candidates</h1>",
        "<p>Each video follows 5,000 warmup steps with 1,000 more native "
        "simulation steps. Frames are enlarged for display. Robust status "
        "requires the catalog's perturbation and localization tests.</p>",
        '<label>Filter gallery <input id="query" type="search" '
        'placeholder="source, index, category"></label><main>',
    ]
    for c in cards:
        search = f"{c['source_kind']} {c['source_id']} {c['category']}".lower()
        html.append(
            f'<article data-search="{search}"><video controls preload="none" poster="{c["poster"]}" '
            f'src="{c["video"]}"></video><p>#{c["rank"]} · '
            f"{c['source_kind']} {c['source_id']} · {c['category']} · speed "
            f"{c['speed']:.3f} px/step · damage {c['damage']:.2f} · "
            f"noise {c['noise']:.2f} · robust {c['robust']}</p></article>"
        )
    html.append(
        '</main><script>document.querySelector("#query").addEventListener('
        '"input", e => { const q=e.target.value.toLowerCase(); '
        'document.querySelectorAll("article").forEach(a => '
        "a.hidden=!a.dataset.search.includes(q)); });</script></html>"
    )
    (args.output / "index.html").write_text("\n".join(html))


if __name__ == "__main__":
    main()
