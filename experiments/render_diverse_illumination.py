#!/usr/bin/env python3
"""Select and replay diverse ASAL illumination members at native world resolution."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(archive, output, count, resolution):
    if resolution < 128 or resolution % 128 or resolution & (resolution - 1):
        raise ValueError("Resolution must be a power-of-two multiple of 128")
    output.mkdir(parents=True, exist_ok=True)
    params = np.load(archive / "parameters.npy", mmap_mode="r")
    z = np.load(archive / "embeddings.npy", mmap_mode="r")
    novelty = np.load(archive / "novelty.npy", mmap_mode="r")
    if len(params) != len(z) or len(z) != len(novelty) or not 1 <= count <= len(z):
        raise ValueError("Archive arrays or requested count are invalid")
    with (archive / "checkpoint.pkl").open("rb") as handle:
        rng = np.asarray(pickle.load(handle)["rng"])
    z = np.asarray(z, dtype=np.float32)
    chosen = [int(np.argmax(novelty))]
    nearest = z @ z[chosen[0]]
    for _ in range(1, count):
        index = int(np.argmin(nearest))
        chosen.append(index)
        nearest = np.maximum(nearest, z @ z[index])
    bundle = output / "selection.npz"
    np.savez(
        bundle, params=np.asarray(params[chosen]), indices=np.asarray(chosen), rng=rng
    )
    metadata = {
        "method": "ASAL archive embedding farthest-point traversal, seeded by maximum nearest-neighbor novelty",
        "count": count,
        "original_archive": str(archive),
        "original_archive_sha256": sha256(archive / "parameters.npy"),
        "original_embedding_sha256": sha256(archive / "embeddings.npy"),
        "bundle_sha256": sha256(bundle),
        "rollout_steps": 256,
        "still_time_point": "state after 256 Lenia updates, matching the last frame of the 16-second videos",
        "world_size": resolution,
        "world_scale": resolution // 128,
        "phenotype_size": resolution,
        "video_size": resolution,
        "fps": 16,
        "render": f"native {resolution}x{resolution} centered world at {resolution // 128}x Lenia spatial resolution",
        "selected": [
            dict(rank=rank + 1, archive_index=index, novelty=float(novelty[index]))
            for rank, index in enumerate(chosen)
        ],
    }
    (output / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {count} members in {bundle}")


def render(output, batch_size):
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    from PIL import Image

    from simulation.lenia import Lenia
    from simulation.trajectory import sampled_frames

    with np.load(output / "selection.npz", allow_pickle=False) as bundle:
        params = np.asarray(bundle["params"])
        indices = np.asarray(bundle["indices"])
        rng = jnp.asarray(bundle["rng"])
    metadata = json.loads((output / "selection.json").read_text())
    resolution = metadata["world_size"]
    substrate = Lenia(
        grid_size=resolution,
        world_scale=metadata["world_scale"],
        center_phenotype=True,
        phenotype_size=resolution,
        start_pattern="5N7KKM",
        clip1=1.0,
    )

    @jax.jit
    def simulate(batch_params, keys):
        return jax.vmap(
            lambda p, key: sampled_frames(
                substrate, key, p, 256, 256, image_size=resolution
            )
        )(batch_params, keys)

    videos = output / "videos"
    videos.mkdir(exist_ok=True)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        pending = [
            i
            for i in range(start, stop)
            if not (videos / f"{i + 1:03d}-archive-{indices[i]:04d}.mp4").exists()
        ]
        if not pending:
            continue
        keys = jnp.stack([jax.random.fold_in(rng, int(indices[i])) for i in pending])
        frames = np.asarray(
            jax.device_get(simulate(jnp.asarray(params[pending]), keys))
        )
        frames = np.uint8(np.clip(frames, 0, 1) * 255)
        for offset, rank in enumerate(pending):
            stem = f"{rank + 1:03d}-archive-{indices[rank]:04d}"
            path = videos / f"{stem}.mp4"
            temp = videos / f"{stem}.tmp.mp4"
            with imageio.get_writer(
                temp, fps=16, codec="libx264", quality=8, macro_block_size=16
            ) as writer:
                for frame in frames[offset]:
                    writer.append_data(frame)
            temp.replace(path)
            Image.fromarray(frames[offset, -1]).save(videos / f"{stem}.png")
            print(
                f"Rendered {rank + 1}/{len(indices)}: archive {indices[rank]}",
                flush=True,
            )
    checksums = {p.name: sha256(p) for p in sorted(videos.glob("*"))}
    (output / "sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")


def verify(output):
    import imageio.v2 as imageio

    metadata = json.loads((output / "selection.json").read_text())
    assert sha256(output / "selection.npz") == metadata["bundle_sha256"]
    checksums = json.loads((output / "sha256.json").read_text())
    videos = output / "videos"
    assert len(checksums) == metadata["count"] * 2
    for name, expected in checksums.items():
        assert sha256(videos / name) == expected, name
    for row in metadata["selected"]:
        stem = f"{row['rank']:03d}-archive-{row['archive_index']:04d}"
        with imageio.get_reader(videos / f"{stem}.mp4") as reader:
            assert reader.count_frames() == 256, stem
            assert reader.get_data(0).shape == (
                metadata["video_size"],
                metadata["video_size"],
                3,
            ), stem
            assert abs(reader.get_meta_data()["fps"] - metadata["fps"]) < 0.01, stem
    print(
        f"Verified {metadata['count']} videos, 256 frames each, "
        f"{metadata['video_size']}x{metadata['video_size']}; "
        f"{len(checksums)} checksums"
    )


def gallery(output):
    metadata = json.loads((output / "selection.json").read_text())
    cards = []
    for row in metadata["selected"]:
        stem = f"{row['rank']:03d}-archive-{row['archive_index']:04d}"
        cards.append(
            f'<article><video controls preload="none" '
            f'poster="videos/{stem}.png" src="videos/{stem}.mp4"></video>'
            f"<p>#{row['rank']} · archive {row['archive_index']} · "
            f"novelty {row['novelty']:.4f}</p></article>"
        )
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Lenia illumination · 100 native 1024² simulations</title>"
        "<style>body{background:#101016;color:#eee;font:16px system-ui;margin:24px}"
        "h1,p{max-width:900px}main{display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(300px,1fr));gap:18px}article{background:#1b1b24;padding:10px}"
        "video{width:100%;aspect-ratio:1;background:#000}article p{margin:8px 0 0}"
        "</style><h1>100 diverse Lenia simulations</h1>"
        "<p>Native 1024×1024 Lenia world, eight times the discovery grid resolution. "
        "Each video shows 256 simulation steps. Selection follows farthest-point "
        "traversal in MobileCLIP2-S2 embedding space.</p><main>"
        + "".join(cards)
        + "</main></html>"
    )
    (output / "index.html").write_text(page)
    print(f"Wrote {output / 'index.html'}")


def render_stills(output, batch_size):
    import jax
    import jax.numpy as jnp
    from PIL import Image

    from simulation.lenia import Lenia
    from simulation.trajectory import final_frame

    metadata = json.loads((output / "selection.json").read_text())
    resolution = metadata["world_size"]
    with np.load(output / "selection.npz", allow_pickle=False) as bundle:
        params = np.asarray(bundle["params"])
        indices = np.asarray(bundle["indices"])
        rng = jnp.asarray(bundle["rng"])
    substrate = Lenia(
        grid_size=resolution,
        world_scale=metadata["world_scale"],
        center_phenotype=True,
        phenotype_size=resolution,
        start_pattern="5N7KKM",
        clip1=1.0,
    )

    @jax.jit
    def simulate(batch_params, keys):
        return jax.vmap(
            lambda p, key: final_frame(substrate, key, p, 255, image_size=resolution)
        )(batch_params, keys)

    stills = output / "stills"
    stills.mkdir(exist_ok=True)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        pending = [
            i
            for i in range(start, stop)
            if not (stills / f"{i + 1:03d}-archive-{indices[i]:04d}.png").exists()
        ]
        if not pending:
            continue
        keys = jnp.stack([jax.random.fold_in(rng, int(indices[i])) for i in pending])
        frames = np.asarray(
            jax.device_get(simulate(jnp.asarray(params[pending]), keys))
        )
        frames = np.uint8(np.clip(frames, 0, 1) * 255)
        for offset, rank in enumerate(pending):
            stem = f"{rank + 1:03d}-archive-{indices[rank]:04d}"
            path = stills / f"{stem}.png"
            temp = stills / f"{stem}.tmp.png"
            Image.fromarray(frames[offset]).save(temp)
            temp.replace(path)
            print(
                f"Rendered still {rank + 1}/{len(indices)}: archive {indices[rank]}",
                flush=True,
            )
    checksums = {p.name: sha256(p) for p in sorted(stills.glob("*.png"))}
    (output / "stills_sha256.json").write_text(json.dumps(checksums, indent=2) + "\n")


def map_stills(output, columns, tile_size):
    import csv

    from PIL import Image
    from scipy.optimize import linear_sum_assignment

    metadata = json.loads((output / "selection.json").read_text())
    selected = metadata["selected"]
    if len(selected) % columns:
        raise ValueError("Selected count must divide evenly into columns")
    rows = len(selected) // columns
    archive = Path(metadata["original_archive"])
    xy = np.load(archive / "umap_coordinates.npy", allow_pickle=False)
    selected_xy = xy[[r["archive_index"] for r in selected]]
    low = np.quantile(selected_xy, 0.01, axis=0)
    high = np.quantile(selected_xy, 0.99, axis=0)
    norm = np.clip((selected_xy - low) / (high - low), 0, 1)
    norm[:, 1] = 1 - norm[:, 1]
    gx, gy = np.meshgrid(
        (np.arange(columns) + 0.5) / columns, (np.arange(rows) + 0.5) / rows
    )
    cells = np.column_stack((gx.ravel(), gy.ravel()))
    cost = ((norm[:, None, :] - cells[None, :, :]) ** 2).sum(axis=2)
    ranks, slots = linear_sum_assignment(cost)
    canvas = Image.new("RGB", (columns * tile_size, rows * tile_size))
    index_rows = []
    for rank, slot in zip(ranks, slots):
        entry = selected[int(rank)]
        x, y = int(slot % columns), int(slot // columns)
        stem = f"{entry['rank']:03d}-archive-{entry['archive_index']:04d}"
        with Image.open(output / "stills" / f"{stem}.png") as image:
            canvas.paste(
                image.convert("RGB").resize(
                    (tile_size, tile_size), Image.Resampling.LANCZOS
                ),
                (x * tile_size, y * tile_size),
            )
        index_rows.append(
            (
                x,
                y,
                entry["rank"],
                entry["archive_index"],
                entry["novelty"],
                float(selected_xy[rank, 0]),
                float(selected_xy[rank, 1]),
                f"stills/{stem}.png",
            )
        )
    canvas.save(output / "map.png", optimize=True)
    with (output / "map_index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "column",
                "row",
                "rank",
                "archive_index",
                "novelty",
                "umap_x",
                "umap_y",
                "native_image",
            )
        )
        writer.writerows(sorted(index_rows, key=lambda row: (row[1], row[0])))
    config = dict(
        columns=columns,
        rows=rows,
        tile_size=tile_size,
        map_size=list(canvas.size),
        source_resolution=metadata["world_size"],
        ordering="minimum squared distance assignment to UMAP grid",
        gutters_px=0,
        map_sha256=sha256(output / "map.png"),
    )
    (output / "map_config.json").write_text(json.dumps(config, indent=2) + "\n")
    index_data = [
        dict(
            column=r[0],
            row=r[1],
            rank=r[2],
            archive_index=r[3],
            novelty=r[4],
            image=r[7],
        )
        for r in sorted(index_rows, key=lambda row: (row[1], row[0]))
    ]
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{len(selected)} diverse Lenia final frames · {metadata['world_size']}²</title>"
        "<style>body{background:#101016;color:#eee;font:16px system-ui;margin:18px}"
        "a{color:#9bd}#scroller{overflow:auto}img{display:block;width:100%;max-width:none;"
        "cursor:crosshair}button{padding:8px;margin:8px 0}</style>"
        f"<h1>{len(selected)} diverse Lenia final frames</h1>"
        "<p>Native 1024×1024 simulations arranged by UMAP position. The "
        f"{columns}×{rows} mosaic has no padding or gaps. Hover for configuration details; "
        "click a tile for its native final frame.</p>"
        '<button id="zoom">Show actual map size</button><span id="info"></span>'
        f'<div id="scroller"><img id="map" src="map.png" alt="Gapless {len(selected)}-tile Lenia map"></div>'
        "<script>const data=" + json.dumps(index_data, separators=(",", ":")) + ";"
        f"const cols={columns},rows={rows},width={canvas.width};"
        'const img=document.getElementById("map"),info=document.getElementById("info");'
        "function tile(e){const r=img.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,"
        "y=(e.clientY-r.top)/r.height;const c=Math.floor(x*cols),row=Math.floor(y*rows);"
        "return data[row*cols+c]}"
        "img.onmousemove=e=>{const t=tile(e);if(t)info.textContent=`  Archive ${t.archive_index}"
        " · rank ${t.rank} · novelty ${Number(t.novelty).toFixed(4)}`};"
        'img.onclick=e=>{const t=tile(e);if(t)window.open(t.image,"_blank")};'
        'document.getElementById("zoom").onclick=()=>{img.style.width='
        'img.style.width===`${width}px`?"100%":`${width}px`};</script></html>'
    )
    (output / "index.html").write_text(page)
    print(
        f"Saved gapless {canvas.width}x{canvas.height} map with {len(selected)} tiles"
    )


def verify_stills(output):
    import csv

    from PIL import Image

    metadata = json.loads((output / "selection.json").read_text())
    assert sha256(output / "selection.npz") == metadata["bundle_sha256"]
    checksums = json.loads((output / "stills_sha256.json").read_text())
    assert len(checksums) == metadata["count"]
    for name, expected in checksums.items():
        path = output / "stills" / name
        assert sha256(path) == expected, name
        with Image.open(path) as image:
            assert image.size == (metadata["world_size"], metadata["world_size"]), name
    config = json.loads((output / "map_config.json").read_text())
    with Image.open(output / "map.png") as image:
        assert list(image.size) == config["map_size"]
    assert sha256(output / "map.png") == config["map_sha256"]
    assert config["gutters_px"] == 0
    with (output / "map_index.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == metadata["count"]
    assert len({(r["column"], r["row"]) for r in rows}) == metadata["count"]
    assert {int(r["archive_index"]) for r in rows} == {
        r["archive_index"] for r in metadata["selected"]
    }
    print(
        f"Verified {metadata['count']} native {metadata['world_size']}² stills "
        f"and gapless {config['map_size'][0]}×{config['map_size'][1]} map"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--resolution", type=int, default=1024)
    r = sub.add_parser("render")
    r.add_argument("output", type=Path)
    r.add_argument("--batch-size", type=int, default=4)
    v = sub.add_parser("verify")
    v.add_argument("output", type=Path)
    g = sub.add_parser("gallery")
    g.add_argument("output", type=Path)
    s = sub.add_parser("render-stills")
    s.add_argument("output", type=Path)
    s.add_argument("--batch-size", type=int, default=4)
    m = sub.add_parser("map-stills")
    m.add_argument("output", type=Path)
    m.add_argument("--columns", type=int, default=25)
    m.add_argument("--tile-size", type=int, default=256)
    vs = sub.add_parser("verify-stills")
    vs.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.archive, args.output, args.count, args.resolution)
    elif args.command == "render":
        render(args.output, args.batch_size)
    elif args.command == "verify":
        verify(args.output)
    elif args.command == "render-stills":
        render_stills(args.output, args.batch_size)
    elif args.command == "map-stills":
        map_stills(args.output, args.columns, args.tile_size)
    elif args.command == "verify-stills":
        verify_stills(args.output)
    else:
        gallery(args.output)


if __name__ == "__main__":
    main()
