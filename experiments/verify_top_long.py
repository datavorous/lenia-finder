#!/usr/bin/env python3
"""Independent 20,000-step control reruns of the strongest catalog entries."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from evaluation.solitons import METRIC_NAMES, Thresholds, make_trial_rollout, summarize
from experiments.verify_solitons import trial_metrics, write_rows
from simulation.lenia import Lenia


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalog", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--seed", type=int, default=202020)
    args = p.parse_args()
    if jax.default_backend() != "gpu":
        p.error("GPU backend required")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.catalog.open(newline="") as f:
        rows = list(csv.DictReader(f))[: args.limit]
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    evaluate = make_trial_rollout(substrate, 20000, "control")
    output_rows = []
    for rank, row in enumerate(rows, 1):
        stem = f"rank-{rank:03d}"
        path = args.output / f"{stem}.npz"
        if path.exists():
            with np.load(path) as z:
                metrics, snapshots = z["metrics"], z["snapshots"]
        else:
            parameter_path = Path(row["parameter_file"])
            if not parameter_path.is_absolute():
                parameter_path = args.catalog.parent / parameter_path
            param = np.load(parameter_path)
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), rank)
            metrics, snapshots = evaluate(jnp.asarray(param[None]), key[None])
            metrics = np.asarray(jax.device_get(metrics))[0]
            snapshots = np.asarray(jax.device_get(snapshots))[0]
            np.savez_compressed(
                path,
                metrics=metrics,
                snapshots=snapshots,
                metric_names=np.asarray(METRIC_NAMES),
            )
        trial = trial_metrics(metrics, snapshots, 5000, Thresholds())
        summary = summarize(metrics, snapshots[-1], 20000)
        output_rows.append(
            dict(
                rank=rank,
                source_kind=row["source_kind"],
                source_id=row["source_id"],
                seed=args.seed,
                **trial,
                **{f"summary_{k}": v for k, v in summary.items()},
            )
        )
        print(
            f"20k control {rank}/{len(rows)}: {row['source_kind']} "
            f"{row['source_id']} {summary['category']}",
            flush=True,
        )
    if output_rows:
        write_rows(args.output / "controls_20000.csv", output_rows)
    config = dict(
        source_catalog=str(args.catalog.resolve()),
        count=len(rows),
        steps=20000,
        sample_period=100,
        snapshot_period=1000,
        seed=args.seed,
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
