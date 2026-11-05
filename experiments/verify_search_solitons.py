#!/usr/bin/env python3
"""Verify promoted discoveries from a neighborhood or MAP-Elites run."""

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


def candidates(search_run, limit):
    rows = []
    for path in (search_run / "promotion_chunks").glob("batch-*.csv"):
        with path.open(newline="") as f:
            rows.extend(r for r in csv.DictReader(f) if r["persistent"] == "True")
    moving = [r for r in rows if r["category"] == "moving_soliton"]
    other = [r for r in rows if r["category"] != "moving_soliton"]

    def spread(pool, count, movement_first=False):
        if not pool or count <= 0:
            return []
        d = np.asarray(
            [
                [
                    float(r[k])
                    for k in (
                        "final_mass_fraction",
                        "final_occupied_fraction",
                        "last_1000_speed_px_per_step",
                        "aligned_activity",
                    )
                ]
                for r in pool
            ],
            np.float64,
        )
        d[:, 2:] = np.log1p(np.maximum(d[:, 2:], 0.0))
        low, high = np.nanquantile(d, [0.05, 0.95], axis=0)
        d = np.clip((d - low) / np.maximum(high - low, 1e-9), 0, 1)
        if movement_first:
            first = int(
                np.nanargmax(
                    d[:, 2] - 0.2 * np.asarray([float(r["mass_cv"]) for r in pool])
                )
            )
        else:
            first = int(np.nanargmin([float(r["mass_cv"]) for r in pool]))
        chosen = [first]
        nearest = np.sum((d - d[first]) ** 2, axis=1)
        for _ in range(1, min(count, len(pool))):
            nearest[chosen] = -1
            index = int(np.argmax(nearest))
            chosen.append(index)
            nearest = np.minimum(nearest, np.sum((d - d[index]) ** 2, axis=1))
        return [pool[i] for i in chosen]

    chosen = spread(moving, limit // 2, True) + spread(
        other, limit - min(len(moving), limit // 2)
    )
    return chosen[:limit]


def proposal(search_run, candidate_id, batch):
    path = search_run / f"promoted-{candidate_id // batch:07d}.npz"
    with np.load(path) as z:
        matching = np.flatnonzero(z["candidate_ids"] == candidate_id)
        if len(matching) != 1:
            raise ValueError(f"Candidate {candidate_id} absent from {path}")
        return np.asarray(z["parameters"][matching[0]], np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("search_run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=515151)
    args = p.parse_args()
    if jax.default_backend() != "gpu":
        p.error("GPU backend required")
    if args.steps < 10000 or args.steps % 1000:
        p.error("steps must be >=10000 and divisible by 1000")
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((args.search_run / "search_config.json").read_text())
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    config.update(
        source_search_config=run_config,
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        intervention_step=5000,
        parameter_logit_std=0.02,
        parameter_scope="dynamic_logits_0_44",
        initial_logit_std=0.10,
        thresholds=Thresholds().to_dict(),
    )
    path = args.output / "verify_config.json"
    if path.exists():
        old_config = json.loads(path.read_text())
        if any(old_config.get(k) != v for k, v in config.items()):
            p.error("Cannot resume with changed configuration")
        config = old_config
    path.write_text(json.dumps(config, indent=2) + "\n")
    rows = candidates(args.search_run, args.limit)
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    modes = ("control", "damage", "noise", "parameter", "initial")
    evaluators = {
        mode: make_trial_rollout(substrate, args.steps, mode) for mode in modes
    }
    thresholds = Thresholds()
    for position, source in enumerate(rows, 1):
        candidate_id = int(source["candidate_id"])
        target = args.output / f"candidate-{candidate_id:08d}"
        if (target / "trials.csv").exists():
            continue
        target.mkdir(exist_ok=True)
        param = proposal(args.search_run, candidate_id, int(run_config["batch"]))
        np.save(target / "parameters.npy", param)
        (target / "source_5000.json").write_text(json.dumps(source, indent=2) + "\n")
        trial_rows = []
        for mode in modes:
            count = 1 if mode == "control" else args.trials
            keys = jnp.stack(
                [
                    jax.random.fold_in(
                        jax.random.PRNGKey(args.seed),
                        candidate_id * 100 + modes.index(mode) * 10 + i,
                    )
                    for i in range(count)
                ]
            )
            series, snapshots = evaluators[mode](
                jnp.asarray(np.repeat(param[None], count, axis=0)), keys
            )
            series = np.asarray(jax.device_get(series))
            snapshots = np.asarray(jax.device_get(snapshots))
            for i in range(count):
                trial = trial_metrics(series[i], snapshots[i], 5000, thresholds)
                summary = summarize(series[i], snapshots[i, -1], args.steps, thresholds)
                trial_rows.append(
                    dict(
                        candidate_id=candidate_id,
                        source_phase=run_config["phase"],
                        mode=mode,
                        trial=i,
                        seed=args.seed,
                        **trial,
                        **{f"summary_{k}": v for k, v in summary.items()},
                    )
                )
            np.savez_compressed(
                target / f"{mode}_trajectories.npz",
                metrics=series,
                snapshots=snapshots,
                metric_names=np.asarray(METRIC_NAMES),
            )
        write_rows(target / "trials.csv", trial_rows)
        print(
            f"search verify {position}/{len(rows)} candidate {candidate_id}: "
            f"control={trial_rows[0]['survived_after_intervention']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
