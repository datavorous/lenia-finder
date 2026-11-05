#!/usr/bin/env python3
"""Long-horizon Lenia controls and damage/noise/parameter/initial trials."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from evaluation.solitons import (
    METRIC_NAMES,
    Thresholds,
    make_trial_rollout,
    periodic_components,
    summarize,
)
from simulation.lenia import Lenia


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def selected_archive_rows(path, limit):
    rows = [r for r in read_csv(path) if r["persistent"] == "True"]
    groups = {}
    for r in rows:
        groups.setdefault(r["category"], []).append(r)
    for group in groups.values():
        group.sort(key=lambda r: (float(r["mass_cv"]), float(r["area_cv"])))
    selected = []
    categories = ("moving_soliton", "oscillator", "static_soliton")
    while len(selected) < limit and any(groups.get(c) for c in categories):
        for c in categories:
            if groups.get(c) and len(selected) < limit:
                selected.append(groups[c].pop(0))
    return selected


def trial_metrics(series, snapshots, intervention_step, thresholds):
    m = np.asarray(series, np.float64)
    mass, area = m[:, 0], m[:, 1]
    late = m[len(m) // 2 :]
    components = [
        periodic_components(
            w,
            thresholds.component_activation_threshold,
            thresholds.component_blur_radius_px,
        )
        for w in snapshots
    ]
    coherent_fraction = float(
        np.mean(
            [
                c["component_count"] == 1
                and c["largest_component_mass_fraction"]
                >= thresholds.min_largest_component_fraction
                for c in components
            ]
        )
    )
    bounded_series = (
        np.isfinite(m).all(axis=1)
        & (mass > thresholds.min_mass_fraction)
        & (mass < thresholds.max_mass_fraction)
        & (area < thresholds.max_occupied_fraction)
    )
    pre_mass = np.nanmean(
        m[max(0, intervention_step // 100 - 10) : intervention_step // 100, 0]
    )
    late_mass = np.nanmean(late[:, 0])
    recovery_ratio = float(late_mass / max(pre_mass, 1e-12))
    late_cv = float(np.nanstd(late[:, 0]) / max(late_mass, 1e-12))
    survived = bool(
        np.all(bounded_series[intervention_step // 100 :])
        and coherent_fraction >= 0.9
        and late_cv <= 0.35
        and 0.5 <= recovery_ratio <= 2.0
    )
    return dict(
        bounded_sample_fraction=float(np.mean(bounded_series)),
        post_bounded_sample_fraction=float(
            np.mean(bounded_series[intervention_step // 100 :])
        ),
        coherent_snapshot_fraction=coherent_fraction,
        minimum_mass_fraction=float(np.nanmin(mass)),
        maximum_mass_fraction=float(np.nanmax(mass)),
        maximum_occupied_fraction=float(np.nanmax(area)),
        late_mass_cv=late_cv,
        recovery_mass_ratio=recovery_ratio,
        survived_after_intervention=survived,
        component_counts=";".join(str(c["component_count"]) for c in components),
    )


def write_rows(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("screen_csv", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=424242)
    args = p.parse_args()
    if jax.default_backend() != "gpu":
        p.error("GPU JAX backend required")
    if args.steps < 10000 or args.steps % 1000:
        p.error("steps must be >=10000 and divisible by 1000")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    config.update(
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        modes=["control", "damage", "noise", "parameter", "initial"],
        intervention_step=5000,
        damage_radius_px=12,
        damage_center_jitter_px=12,
        noise_radius_px=20,
        noise_std=0.10,
        parameter_logit_std=0.02,
        parameter_scope="dynamic_logits_0_44",
        initial_logit_std=0.10,
        component_snapshot_period=1000,
        numerical_sample_period=100,
    )
    conf = args.output / "verify_config.json"
    if conf.exists():
        old_config = json.loads(conf.read_text())
        if any(old_config.get(k) != v for k, v in config.items()):
            p.error("Cannot resume with changed configuration")
        config = old_config
    conf.write_text(json.dumps(config, indent=2) + "\n")
    parameters = np.load(args.archive / "parameters.npy", mmap_mode="r")
    rows = selected_archive_rows(args.screen_csv, args.limit)
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    thresholds = Thresholds()
    modes = ("control", "damage", "noise", "parameter", "initial")
    evaluators = {
        mode: make_trial_rollout(substrate, args.steps, mode) for mode in modes
    }
    for position, source in enumerate(rows):
        index = int(source["archive_index"])
        target = args.output / f"archive-{index:04d}"
        result_path = target / "trials.csv"
        if result_path.exists():
            continue
        target.mkdir(exist_ok=True)
        param = np.asarray(parameters[index], np.float32)
        np.save(target / "parameters.npy", param)
        trial_rows = []
        for mode in modes:
            count = 1 if mode == "control" else args.trials
            keys = jnp.stack(
                [
                    jax.random.fold_in(
                        jax.random.PRNGKey(args.seed),
                        index * 1000 + modes.index(mode) * 100 + i,
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
                        archive_index=index,
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
        write_rows(result_path, trial_rows)
        print(
            f"verified {position + 1}/{len(rows)} index {index}: "
            f"control={trial_rows[0]['survived_after_intervention']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
