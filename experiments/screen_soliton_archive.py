#!/usr/bin/env python3
"""Resumable, sharded numerical screening of the 5,000-member Lenia archive."""

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

if (
    "--device=cpu" in sys.argv
    or "--device" in sys.argv
    and sys.argv[sys.argv.index("--device") + 1] == "cpu"
):
    os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from evaluation.solitons import METRIC_NAMES, Thresholds, make_rollout, summarize
from simulation.lenia import Lenia


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def run(args):
    archive = args.archive.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    params = np.load(archive / "parameters.npy", mmap_mode="r")
    if params.ndim != 2 or params.shape[1] != 3117:
        raise ValueError("Expected the ASAL-derived 3,117-dimensional Lenia archive")
    thresholds = Thresholds()
    stages = tuple(int(s) for s in args.stages.split(","))
    if stages != tuple(sorted(stages)) or any(s % args.sample_period for s in stages):
        raise ValueError("Stages must increase and divide by sample-period")
    indices = np.arange(len(params))[args.shard_index :: args.num_shards]
    if args.max_configs:
        indices = indices[: args.max_configs]
    config = dict(
        archive=str(archive),
        archive_sha256=digest(archive / "parameters.npy"),
        archive_config=json.loads((archive / "config.json").read_text()),
        seed=args.seed,
        indices=indices.tolist(),
        stages=stages,
        sample_period=args.sample_period,
        batch=args.batch,
        thresholds=thresholds.to_dict(),
        threshold_sources={
            "mass_upper": "CARL 2026: 10% after 5,000 steps; adapted to channel-mean fraction",
            "phase_ratio": "Hamon 2025 reference code: (0,2) last quarter / first quarter",
            "components": "Hamon 2025 reference code: one blurred connected object",
            "movement": "Hamon 2025 reference code: 100 px over 1,000 steps on 256 grid; scaled to 50 px on 128 grid",
            "cv_area_activity": "implementation-calibrated; not paper thresholds",
        },
        grid_size=128,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
        world_scale=1,
        python=platform.python_version(),
        jax=jax.__version__,
        backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],
    )
    config_path = output / "screen_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in (
            "archive_sha256",
            "seed",
            "indices",
            "stages",
            "sample_period",
            "batch",
            "thresholds",
        ):
            if old[key] != config[key]:
                raise ValueError(f"Cannot resume with changed {key}")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    for stage in stages:
        stage_dir = output / f"stage-{stage:05d}"
        stage_dir.mkdir(exist_ok=True)
        evaluate = make_rollout(substrate, stage, args.sample_period)
        rows = []
        stage_started = time.monotonic()
        for start in range(0, len(indices), args.batch):
            stop = min(start + args.batch, len(indices))
            subset = indices[start:stop]
            stem = f"batch-{start:05d}-{stop:05d}"
            summary_path = stage_dir / f"{stem}.json"
            metrics_path = stage_dir / f"{stem}.npz"
            if summary_path.exists() and metrics_path.exists():
                batch_rows = json.loads(summary_path.read_text())
            else:
                keys = jnp.stack(
                    [
                        jax.random.fold_in(jax.random.PRNGKey(args.seed), int(i))
                        for i in subset
                    ]
                )
                values, worlds = evaluate(jnp.asarray(params[subset]), keys)
                values = np.asarray(jax.device_get(values))
                worlds = np.asarray(jax.device_get(worlds))
                batch_rows = []
                for offset, index in enumerate(subset):
                    result = summarize(
                        values[offset], worlds[offset], stage, thresholds
                    )
                    batch_rows.append(
                        dict(archive_index=int(index), seed=args.seed, **result)
                    )
                temp = metrics_path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    temp,
                    archive_indices=subset,
                    metrics=values,
                    metric_names=np.asarray(METRIC_NAMES),
                )
                temp.replace(metrics_path)
                temp_json = summary_path.with_suffix(".tmp")
                temp_json.write_text(json.dumps(batch_rows))
                temp_json.replace(summary_path)
            rows.extend(batch_rows)
            if (start // args.batch) % 10 == 0 or stop == len(indices):
                print(
                    f"stage {stage}: {stop}/{len(indices)}; "
                    f"bounded {sum(bool(r['bounded']) for r in rows)}; "
                    f"{time.monotonic() - stage_started:.1f}s",
                    flush=True,
                )
        rows.sort(key=lambda row: row["archive_index"])
        write_csv(output / f"stage-{stage:05d}.csv", rows)
        (output / f"stage-{stage:05d}-counts.json").write_text(
            json.dumps(
                dict(
                    evaluated=len(rows),
                    bounded=sum(bool(r["bounded"]) for r in rows),
                    persistent=sum(bool(r["persistent"]) for r in rows),
                    categories={
                        cat: sum(r["category"] == cat for r in rows)
                        for cat in sorted({r["category"] for r in rows})
                    },
                ),
                indent=2,
            )
            + "\n"
        )
        if stage != stages[-1]:
            survivors = {r["archive_index"] for r in rows if r["bounded"]}
            indices = np.asarray([i for i in indices if i in survivors], dtype=np.int32)
            print(
                f"Advancing {len(indices)} bounded configurations to next stage",
                flush=True,
            )
        if not len(indices):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stages", default="500,1000,5000")
    parser.add_argument("--sample-period", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.device == "gpu" and jax.default_backend() != "gpu":
        parser.error("CUDA JAX device is required for GPU screening")
    if not 0 <= args.shard_index < args.num_shards or args.batch < 1:
        parser.error("invalid shard or batch arguments")
    run(args)


if __name__ == "__main__":
    main()
