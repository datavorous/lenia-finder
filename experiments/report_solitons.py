#!/usr/bin/env python3
"""Build a searchable, deduplicated Lenia soliton catalog and figures."""

import argparse
import csv
import json
import math
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_trials(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def bool_value(value):
    return str(value).lower() == "true"


def control_radius_cv(folder):
    with np.load(folder / "control_trajectories.npz") as z:
        radius = z["metrics"][0, :, 4].astype(np.float64)
    late = radius[len(radius) // 2 :]
    return float(np.nanstd(late) / max(np.nanmean(late), 1e-12))


def archive_morphology(screen_csv, frame):
    source = screen_csv.parent / "stage-05000"
    values = {}
    if source.exists():
        for file in sorted(source.glob("batch-*.npz")):
            with np.load(file) as z:
                metrics = z["metrics"]
                indices = z["archive_indices"]
            radius = metrics[:, metrics.shape[1] // 2 :, 4]
            cv = np.nanstd(radius, axis=1) / np.maximum(
                np.nanmean(radius, axis=1), 1e-12
            )
            values.update(zip((int(i) for i in indices), (float(v) for v in cv)))
    frame["radius_cv"] = frame["archive_index"].map(values)
    frame["morphology_stable"] = frame["radius_cv"] < 0.20
    return frame


def deduplicate(rows, output):
    if len(rows) < 2:
        return list(range(len(rows)))
    controls = []
    images = []
    for r in rows:
        folder = (output / r["verification_folder"]).resolve()
        with np.load(folder / "control_trajectories.npz") as z:
            m = z["metrics"][0].astype(np.float64)
            world = z["snapshots"][0, -1].astype(np.float32)
        controls.append(m)
        # Centered native world, downsampled only for duplicate comparison.
        small = world.reshape(32, 4, 32, 4, 3).mean((1, 3)).ravel()
        images.append(small / max(np.linalg.norm(small), 1e-9))
    desc = np.array(
        [
            [
                float(r[k])
                for k in (
                    "final_mass_fraction",
                    "final_occupied_fraction",
                    "mass_cv",
                    "area_cv",
                    "aligned_activity",
                    "last_1000_speed_px_per_step",
                )
            ]
            for r in rows
        ]
    )
    desc[:, -1] = np.log1p(np.maximum(desc[:, -1], 0.0))
    desc[:, -2] = np.log1p(np.maximum(desc[:, -2], 0.0))
    med = np.nanmedian(desc, axis=0)
    mad = np.nanmedian(np.abs(desc - med), axis=0)
    desc = (desc - med) / np.maximum(mad * 4, 0.02)
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i):
            descriptor_distance = np.linalg.norm(desc[i] - desc[j])
            image_cosine = float(np.dot(images[i], images[j]))
            a, b = controls[i], controls[j]
            trajectory_difference = np.nanmean(np.abs(a[:, :2] - b[:, :2]))
            if (
                descriptor_distance < 0.75
                and image_cosine > 0.98
                and trajectory_difference < 0.01
            ):
                parent[root(i)] = root(j)
    roots = [root(i) for i in range(len(rows))]
    index = {r: n for n, r in enumerate(sorted(set(roots)))}
    return [index[r] for r in roots]


def build_catalog(screen, verify_root, search_verify_roots, output, long_controls=None):
    screen_by_index = {int(r["archive_index"]): r for r in read_trials(screen)}
    rows = []
    for folder in sorted(verify_root.glob("archive-*")):
        trials_file = folder / "trials.csv"
        if not trials_file.exists():
            continue
        trials = read_trials(trials_file)
        index = int(folder.name.split("-")[1])
        base = screen_by_index[index].copy()
        base["archive_index"] = index
        base["source_kind"] = "visual_archive"
        base["source_id"] = index
        base["verification_folder"] = os.path.relpath(folder, output)
        control = next(t for t in trials if t["mode"] == "control")
        base["control_10000_survived"] = bool_value(
            control["survived_after_intervention"]
        )
        base["control_bounded_sample_fraction"] = float(
            control["bounded_sample_fraction"]
        )
        base["control_coherent_snapshot_fraction"] = float(
            control["coherent_snapshot_fraction"]
        )
        base["control_localized"] = bool(
            float(control["summary_max_border_mass_fraction"]) < 0.10
            and float(control["summary_final_radius_px"]) < 40.0
        )
        base["control_radius_cv"] = control_radius_cv(folder)
        base["control_morphology_stable"] = base["control_radius_cv"] < 0.20
        for mode in ("damage", "noise", "parameter", "initial"):
            subset = [t for t in trials if t["mode"] == mode]
            base[f"{mode}_survival_fraction"] = float(
                np.mean([bool_value(t["survived_after_intervention"]) for t in subset])
            )
            base[f"{mode}_trial_count"] = len(subset)
        base["damage_noise_survival_fraction"] = (
            base["damage_survival_fraction"] + base["noise_survival_fraction"]
        ) / 2
        base["parameter_file"] = os.path.relpath(folder / "parameters.npy", output)
        base["trial_file"] = os.path.relpath(trials_file, output)
        rows.append(base)
    for source_kind, root in search_verify_roots.items():
        for folder in sorted(root.glob("candidate-*")):
            trials_file = folder / "trials.csv"
            source_file = folder / "source_5000.json"
            if not trials_file.exists() or not source_file.exists():
                continue
            trials = read_trials(trials_file)
            base = json.loads(source_file.read_text())
            candidate_id = int(base["candidate_id"])
            base["archive_index"] = -1
            base["source_kind"] = source_kind
            base["source_id"] = candidate_id
            base["verification_folder"] = os.path.relpath(folder, output)
            control = next(t for t in trials if t["mode"] == "control")
            base["control_10000_survived"] = bool_value(
                control["survived_after_intervention"]
            )
            base["control_bounded_sample_fraction"] = float(
                control["bounded_sample_fraction"]
            )
            base["control_coherent_snapshot_fraction"] = float(
                control["coherent_snapshot_fraction"]
            )
            base["control_localized"] = bool(
                float(control["summary_max_border_mass_fraction"]) < 0.10
                and float(control["summary_final_radius_px"]) < 40.0
            )
            base["control_radius_cv"] = control_radius_cv(folder)
            base["control_morphology_stable"] = base["control_radius_cv"] < 0.20
            for mode in ("damage", "noise", "parameter", "initial"):
                subset = [t for t in trials if t["mode"] == mode]
                base[f"{mode}_survival_fraction"] = float(
                    np.mean(
                        [bool_value(t["survived_after_intervention"]) for t in subset]
                    )
                )
                base[f"{mode}_trial_count"] = len(subset)
            base["damage_noise_survival_fraction"] = (
                base["damage_survival_fraction"] + base["noise_survival_fraction"]
            ) / 2
            base["parameter_file"] = os.path.relpath(folder / "parameters.npy", output)
            base["trial_file"] = os.path.relpath(trials_file, output)
            rows.append(base)
    if not rows:
        return []
    clusters = deduplicate(rows, output)
    for r, cluster in zip(rows, clusters):
        r["duplicate_cluster"] = cluster
        r["perturbation_robust"] = bool(
            r["control_10000_survived"]
            and r["damage_survival_fraction"] >= 0.75
            and r["noise_survival_fraction"] >= 0.75
            and r["parameter_survival_fraction"] >= 0.75
            and r["initial_survival_fraction"] >= 0.75
        )
        r["dynamically_active"] = bool_value(r["active"])
        r["robust_verified"] = bool(
            r["perturbation_robust"]
            and r["dynamically_active"]
            and r["control_localized"]
            and r["control_morphology_stable"]
        )
    controls = {}
    if long_controls and long_controls.exists():
        for record in read_trials(long_controls):
            controls[(record["source_kind"], int(record["source_id"]))] = record
    for r in rows:
        record = controls.get((r["source_kind"], int(r["source_id"])))
        r["control_20000_survived"] = (
            (
                bool_value(record["survived_after_intervention"])
                and bool_value(record["summary_persistent"])
                and bool_value(record.get("summary_morphology_stable", "True"))
            )
            if record
            else None
        )
        if record:
            r["robust_verified"] = bool(
                r["robust_verified"] and r["control_20000_survived"]
            )
    rows.sort(
        key=lambda r: (
            r["robust_verified"],
            r["damage_noise_survival_fraction"],
            r["control_10000_survived"],
            float(r["last_1000_speed_px_per_step"]),
            -float(r["mass_cv"]),
        ),
        reverse=True,
    )
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "catalog.csv", index=False)
    frame.to_parquet(output / "catalog.parquet", index=False)
    with sqlite3.connect(output / "catalog.sqlite") as db:
        frame.to_sql("discoveries", db, if_exists="replace", index=False)
        db.execute(
            "CREATE INDEX IF NOT EXISTS discovery_source ON discoveries(source_kind, source_id)"
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS discovery_class ON discoveries(category, robust_verified)"
        )
    (output / "catalog.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def search_counts(path):
    chunks = path / "candidate_chunks"
    promotions = path / "promotion_chunks"
    if not chunks.exists():
        return None
    candidates = bounded = promoted = persistent = moving = 0
    categories = Counter()
    for file in sorted(chunks.glob("batch-*.csv")):
        for r in read_trials(file):
            candidates += 1
            bounded += bool_value(r["bounded"])
    for file in sorted(promotions.glob("batch-*.csv")):
        for r in read_trials(file):
            promoted += 1
            persistent += bool_value(r["persistent"])
            moving += r["category"] == "moving_soliton"
            categories[r["category"]] += 1
    cells = 0
    state = (
        (path / "state-repaired.npz")
        if (path / "state-repaired.npz").exists()
        else path / "state.npz"
    )
    if state.exists():
        with np.load(state) as z:
            cells = len(z["elite_cells"])
    return dict(
        candidates=candidates,
        short_bounded=bounded,
        promoted_5000=promoted,
        long_persistent=persistent,
        long_moving=moving,
        occupied_cells=cells,
        total_steps=candidates * 100 + promoted * 5000,
        categories=dict(categories),
    )


def cma_counts(path):
    if not path or not path.exists():
        return None
    config_file = path / "refine_config.json"
    if not config_file.exists():
        return None
    config = json.loads(config_file.read_text())
    records = []
    for file in path.glob("archive-*/generations.csv"):
        records.extend(read_trials(file))
    return dict(
        seeds=len(list(path.glob("archive-*/generations.csv"))),
        generations=len(records),
        short_evaluations=len(records) * int(config["population"]),
        long_evaluations=len(records) * int(config["population"]),
        long_persistent=sum(bool_value(r["long_persistent"]) for r in records),
        long_moving=sum(r["long_category"] == "moving_soliton" for r in records),
        population_long_persistent=sum(
            int(r.get("population_persistent", 0)) for r in records
        ),
        total_steps=len(records) * int(config["population"]) * 5500,
    )


def figures(screen, catalog, searches, output):
    frame = pd.read_csv(screen)
    if "survival_duration_steps" not in frame:
        frame["survival_duration_steps"] = np.where(
            frame["first_failure_sample"] < 0,
            frame["steps"],
            (frame["first_failure_sample"] + 1) * 100,
        )
    plt.rcParams.update({"figure.figsize": (7, 4), "font.size": 10})
    fig, ax = plt.subplots(3, 2, figsize=(11, 11))
    fields = (
        ("survival_duration_steps", "Survival duration (steps)"),
        ("final_occupied_fraction", "Occupied fraction"),
        ("last_1000_speed_px_per_step", "Speed (px/step)"),
        ("mass_cv", "Late mass CV"),
        ("final_mass_fraction", "Mass fraction"),
        ("area_cv", "Late area CV"),
    )
    for a, (field, title) in zip(ax.flat, fields):
        vals = pd.to_numeric(frame[field], errors="coerce")
        vals = vals[np.isfinite(vals)]
        a.hist(vals, bins=40, color="#4d80af")
        a.set_title(title)
        a.set_ylabel("Configurations")
    fig.tight_layout()
    fig.savefig(output / "archive_distributions.png", dpi=160)
    plt.close(fig)
    if catalog:
        fig, ax = plt.subplots(figsize=(8, 4))
        data = [
            [float(r[f"{mode}_survival_fraction"]) for r in catalog]
            for mode in ("damage", "noise", "parameter", "initial")
        ]
        ax.boxplot(data, tick_labels=["Damage", "Noise", "Parameter", "Initial"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Fraction of trials surviving and recovering")
        ax.set_title("10,000-step verification")
        fig.tight_layout()
        fig.savefig(output / "robustness_distributions.png", dpi=160)
        plt.close(fig)


def threshold_sensitivity(frame, searches, output):
    rows = []
    for max_area in (0.10, 0.15, 0.20, 0.25, 0.30):
        for max_mass_cv in (0.20, 0.35, 0.50):
            mask = (
                (frame["final_mass_fraction"] > 1e-6)
                & (frame["final_mass_fraction"] < 0.10)
                & (frame["final_occupied_fraction"] < max_area)
                & (frame["mass_cv"] < max_mass_cv)
                & (frame["area_cv"] < 0.45)
                & (frame["component_count"] == 1)
                & (frame["largest_component_mass_fraction"] >= 0.90)
                & (frame["phase_mass_ratio"] > 0)
                & (frame["phase_mass_ratio"] < 2)
            )
            rows.append(
                dict(
                    max_occupied_fraction=max_area,
                    max_mass_cv=max_mass_cv,
                    passing=int(mask.sum()),
                    passing_moving=int(
                        (
                            mask
                            & frame["moving_sustained"]
                            & frame["moving_hamon_scaled"]
                        ).sum()
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(output / "threshold_sensitivity.csv", index=False)
    table = pd.DataFrame(rows).pivot(
        index="max_mass_cv", columns="max_occupied_fraction", values="passing"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(table.columns)), [f"{v:.2f}" for v in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{v:.2f}" for v in table.index])
    ax.set_xlabel("Maximum occupied fraction")
    ax.set_ylabel("Maximum mass CV")
    ax.set_title("Sensitivity of 5,000-step persistence screen")
    fig.colorbar(im, ax=ax, label="Passing configurations")
    fig.tight_layout()
    fig.savefig(output / "threshold_sensitivity.png", dpi=160)
    plt.close(fig)
    labels, cover, quality = [], [], []
    for label, path in searches.items():
        state = (
            (path / "state-repaired.npz")
            if (path / "state-repaired.npz").exists()
            else path / "state.npz"
        )
        if not state.exists():
            continue
        with np.load(state) as z:
            cells = z["elite_cells"]
            scores = z["elite_quality"]
        grid = np.full((8, 8), np.nan)
        for cell, score in zip(cells, scores):
            y, x = int(cell[0]), int(cell[1])
            grid[y, x] = (
                np.nanmax([grid[y, x], score]) if np.isfinite(grid[y, x]) else score
            )
        labels.append(label)
        cover.append(len(cells))
        quality.append(grid)
    if quality:
        fig, axes = plt.subplots(1, len(quality), figsize=(5 * len(quality), 4))
        axes = np.atleast_1d(axes)
        for a, label, grid, count in zip(axes, labels, quality, cover):
            im = a.imshow(grid, origin="lower", vmin=0, vmax=1.2, cmap="viridis")
            a.set_title(f"{label}: {count} occupied 4D cells")
            a.set_xlabel("Occupied area bin")
            a.set_ylabel("Mass bin")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="Best short score")
        fig.savefig(output / "map_elites_coverage.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def coverage_curves(searches, output):
    paths = []
    if searches.get("neighborhood"):
        paths.append(("guided", searches["neighborhood"]))
    if searches.get("QD"):
        paths.append(("guided", searches["QD"]))
    if searches.get("random"):
        paths.append(("random", searches["random"]))
    maps = {"guided": {}, "random": {}}
    counts = {"guided": 0, "random": 0}
    records = []
    for method, path in paths:
        for chunk in sorted((path / "candidate_chunks").glob("batch-*.csv")):
            with chunk.open(newline="") as f:
                for row in csv.DictReader(f):
                    counts[method] += 1
                    if bool_value(row["bounded"]):
                        values = (
                            float(row["mass"]) / 0.1,
                            float(row["area"]) / 0.2,
                            float(row["displacement_1000_extrapolated"]) / 100.0,
                            float(row["activity"]) / 2.0,
                        )
                        bins = [
                            min(7, max(0, int(v * 8)))
                            if math.isfinite(v)
                            else (7 if v > 0 else 0)
                            for v in values
                        ]
                        cell = "-".join(map(str, bins))
                        score = float(row["quality"])
                        maps[method][cell] = max(maps[method].get(cell, -np.inf), score)
                    if counts[method] % 10000 == 0:
                        records.append(
                            dict(
                                method=method,
                                evaluations=counts[method],
                                coverage=len(maps[method]),
                                qd_score=sum(
                                    max(v, 0.0) for v in maps[method].values()
                                ),
                            )
                        )
        if counts[method]:
            records.append(
                dict(
                    method=method,
                    evaluations=counts[method],
                    coverage=len(maps[method]),
                    qd_score=sum(max(v, 0.0) for v in maps[method].values()),
                )
            )
    if not records:
        return
    pd.DataFrame(records).to_csv(output / "coverage_curve.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in ("guided", "random"):
        group = [r for r in records if r["method"] == method]
        if not group:
            continue
        x = np.asarray([r["evaluations"] for r in group]) / 1e6
        axes[0].plot(x, [r["coverage"] for r in group], label=method)
        axes[1].plot(x, [r["qd_score"] for r in group], label=method)
    axes[0].set_ylabel("Occupied 4D behavior cells")
    axes[1].set_ylabel("Sum of best short scores")
    for ax in axes:
        ax.set_xlabel("Million candidate evaluations")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "coverage_quality.png", dpi=160)
    plt.close(fig)


def comparison_figure(counts, output):
    archive_steps = counts["archive_evaluated"] * 5000
    items = [
        (
            "Visual archive",
            counts["archive_persistent"],
            counts["archive_moving"],
            archive_steps,
        )
    ]
    guided = [counts["searches"].get(k) for k in ("neighborhood", "QD")]
    guided = [x for x in guided if x]
    if guided:
        items.append(
            (
                "Guided search",
                sum(x["long_persistent"] for x in guided),
                sum(x["long_moving"] for x in guided),
                sum(x["total_steps"] for x in guided),
            )
        )
    random = counts["searches"].get("random")
    if random:
        items.append(
            (
                "Random baseline",
                random["long_persistent"],
                random["long_moving"],
                random["total_steps"],
            )
        )
    if len(items) < 2:
        return
    names = [x[0] for x in items]
    persistent = [x[1] / x[3] * 1e6 for x in items]
    moving = [x[2] / x[3] * 1e6 for x in items]
    x = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, persistent, 0.36, label="5,000-step persistent")
    ax.bar(x + 0.18, moving, 0.36, label="5,000-step moving")
    ax.set_xticks(x, names)
    ax.set_ylabel("Passing evaluations per million simulated steps")
    ax.set_title("Observed screening yield by search source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "search_comparison.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("screen_csv", type=Path)
    p.add_argument("verify_root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--neighborhood", type=Path)
    p.add_argument("--qd", type=Path)
    p.add_argument("--random", type=Path)
    p.add_argument("--verified-neighborhood", type=Path)
    p.add_argument("--verified-qd", type=Path)
    p.add_argument("--cma", type=Path)
    p.add_argument("--gpu-hourly", type=float)
    p.add_argument("--instance-start-utc")
    p.add_argument("--long-controls", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_roots = {
        k: v
        for k, v in (
            ("neighborhood", args.verified_neighborhood),
            ("QD", args.verified_qd),
        )
        if v
    }
    catalog = build_catalog(
        args.screen_csv, args.verify_root, verify_roots, args.output, args.long_controls
    )
    search_paths = {
        k: v
        for k, v in (
            ("neighborhood", args.neighborhood),
            ("QD", args.qd),
            ("random", args.random),
        )
        if v
    }
    searches = {k: search_counts(v) for k, v in search_paths.items()}
    figures(args.screen_csv, catalog, search_paths, args.output)
    screen = pd.read_csv(args.screen_csv)
    screen = archive_morphology(args.screen_csv, screen)
    if "survival_duration_steps" not in screen:
        screen["survival_duration_steps"] = np.where(
            screen["first_failure_sample"] < 0,
            screen["steps"],
            (screen["first_failure_sample"] + 1) * 100,
        ).astype(int)
    screen.to_csv(args.output / "archive_screen.csv", index=False)
    screen.to_parquet(args.output / "archive_screen.parquet", index=False)
    threshold_sensitivity(screen, search_paths, args.output)
    coverage_curves(search_paths, args.output)
    counts = dict(
        archive_evaluated=len(screen),
        archive_bounded=int(screen["bounded"].sum()),
        archive_persistent=int(screen["persistent"].sum()),
        archive_moving=int((screen["category"] == "moving_soliton").sum()),
        archive_categories=screen["category"].value_counts().to_dict(),
        verified_candidates=len(catalog),
        robust_verified=sum(r["robust_verified"] for r in catalog),
        extended_controls=sum(r["control_20000_survived"] is not None for r in catalog),
        extended_control_passes=sum(
            r["control_20000_survived"] is True for r in catalog
        ),
        unique_robust_clusters=len(
            {r["duplicate_cluster"] for r in catalog if r["robust_verified"]}
        ),
        searches=searches,
        verified_by_source=dict(Counter(r["source_kind"] for r in catalog)),
        robust_by_source=dict(
            Counter(r["source_kind"] for r in catalog if r["robust_verified"])
        ),
        trial_survival_by_mode={
            mode: float(np.mean([r[f"{mode}_survival_fraction"] for r in catalog]))
            if catalog
            else None
            for mode in ("damage", "noise", "parameter", "initial")
        },
        cma_refinement=cma_counts(args.cma),
        generated_utc=datetime.now(timezone.utc).isoformat(),
    )
    if args.gpu_hourly and args.instance_start_utc:
        started = datetime.fromisoformat(args.instance_start_utc.replace("Z", "+00:00"))
        elapsed_hours = (datetime.now(timezone.utc) - started).total_seconds() / 3600
        counts["vast_cost_estimate"] = dict(
            hourly_usd=args.gpu_hourly,
            elapsed_hours=elapsed_hours,
            estimated_usd=elapsed_hours * args.gpu_hourly,
            estimate_through_utc=counts["generated_utc"],
            note="Instance rental estimate through report generation; excludes variable network charges",
        )
    comparison_figure(counts, args.output)
    (args.output / "summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    lines = [
        "# Numerical Lenia soliton search",
        "",
        "The simulator is the unchanged ASAL-derived three-channel Lenia",
        "at 128×128 with pattern 5N7KKM and its native timestep.",
        "",
        "## Screening",
        "",
        f"All {len(screen):,} archived configurations were run for 5,000 steps.",
        f"{counts['archive_bounded']:,} were bounded; "
        f"{counts['archive_persistent']:,} passed the calibrated persistence "
        f"and coherence screen; {counts['archive_moving']:,} passed the "
        "sustained movement rule.",
        "",
        "These are numerical classifications, not claims of life or novelty.",
        "",
        "Survival duration is the sampled time to the first mass or area "
        "bound violation, capped at 5,000 steps; it does not estimate "
        "behavior beyond the measured horizon.",
        "",
        "## Verification",
        "",
        f"{len(catalog):,} candidates have 10,000-step controls and "
        "damage, noise, parameter, and initial-condition trials.",
        f"{counts['robust_verified']:,} passed the prespecified 75% survival "
        "fraction in each perturbation family and the separate activity, "
        "morphology, and localization checks; "
        f"{counts['unique_robust_clusters']:,} duplicate clusters remain.",
        "",
        f"Of the ranked candidates, {counts['extended_controls']:,} have "
        f"20,000-step controls and {counts['extended_control_passes']:,} "
        "pass that extended control. A failed extended control removes "
        "a candidate from the robust ranking.",
        "",
        "Mean trial survival fractions: "
        + ", ".join(
            f"{mode} {counts['trial_survival_by_mode'][mode]:.2f}"
            for mode in ("damage", "noise", "parameter", "initial")
        )
        if catalog
        else "No completed perturbation trials yet.",
        "",
        "## Strongest verified configurations",
        "",
        "These are distinct behavioral clusters ranked by the catalog's "
        "numerical recovery and movement criteria. Every row has its "
        "parameters and trial trajectories in the linked catalog.",
        "",
        "| Rank | Source | Class | Speed (px/step) | Damage | Noise | "
        "Parameter | Initial | 20,000 steps |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    strongest = []
    seen_clusters = set()
    for row in catalog:
        if not row["robust_verified"] or row["duplicate_cluster"] in seen_clusters:
            continue
        seen_clusters.add(row["duplicate_cluster"])
        strongest.append(row)
        if len(strongest) == 10:
            break
    for row in strongest:
        extended = row["control_20000_survived"]
        extended_text = (
            "pass" if extended is True else ("fail" if extended is False else "pending")
        )
        lines.append(
            f"| {row['rank']} | {row['source_kind']} {row['source_id']} | "
            f"{row['category']} | {float(row['last_1000_speed_px_per_step']):.3f} | "
            f"{row['damage_survival_fraction']:.2f} | "
            f"{row['noise_survival_fraction']:.2f} | "
            f"{row['parameter_survival_fraction']:.2f} | "
            f"{row['initial_survival_fraction']:.2f} | {extended_text} |"
        )
    if not strongest:
        lines.append(
            "| — | No configuration passed all verification criteria | — | — | — | — | — | — | — |"
        )
    lines += [
        "",
        "## Failure cases",
        "",
        f"At 5,000 steps, {counts['archive_categories'].get('extinct', 0):,} "
        "archive configurations were extinct and "
        f"{counts['archive_categories'].get('expanding', 0):,} exceeded the "
        "mass or occupied-area bounds. Passing the CARL-style mass filter "
        "alone does not imply coherence, persistent morphology, movement, "
        "or recovery from perturbation. Per-trial failures and continuous "
        "measurements are retained in each candidate's trials.csv.",
        "",
        "Damage removes a radius-12 disk at step 5,000; noise adds clipped "
        "Gaussian noise (SD 0.10) in a radius-20 disk. Parameter and "
        "initial-condition trials change separate genotype slices. Recovery "
        "requires bounded mass and area throughout the post-intervention "
        "trajectory, at least 90% coherent snapshots, late mass CV ≤0.35, "
        "and recovered mass 0.5–2× the pre-intervention mass. The original "
        "Lenia initializer ignores its PRNG key, so alternative initial "
        "conditions are produced by perturbing the 3,072 initial-cell "
        "logits; changing a key alone would be an invalid robustness test.",
        "",
        "## Threshold provenance",
        "",
        "- CARL: final mass >0 and <10% after 5,000 steps; here mass is "
        "normalized over the three-channel board.",
        "- Hamon et al.: one blurred connected object, quarter-to-quarter "
        "mass ratio in (0,2), and >100 px displacement over 1,000 steps "
        "on a 256 grid. The movement distance is scaled to 50 px on 128.",
        "- Occupancy <20%, mass CV <0.35, area CV <0.45, component mass "
        "share ≥90%, activity threshold 0.005, and perturbation recovery "
        "thresholds are implementation calibrations, not paper values.",
        "- The final robust catalog additionally requires maximum border "
        "mass fraction <0.10 and final centered radius <40 px in the "
        "10,000-step control. These are implementation localization "
        "calibrations and are reported separately from the archive screen.",
        "- Radius CV <0.20 over the latter half of the control is a "
        "separate morphology-stability calibration. Radius CV for all "
        "5,000 archived configurations is in archive_screen.csv/Parquet.",
        "- threshold_sensitivity.csv and threshold_sensitivity.png show "
        "how the persistence count changes with occupied-area and mass-CV "
        "cutoffs; the main screen uses 0.20 and 0.35, respectively.",
        "- Because metrics are sampled every 100 updates, the displacement "
        "windows span sampled steps 100–1,000 and 4,100–5,000 (900 updates); "
        "the 50-pixel rule is consequently conservative relative to Hamon's "
        "1,000-step window.",
        "",
        "## Search comparison",
        "",
    ]
    for label, data in searches.items():
        if data:
            lines.append(
                f"- {label}: {data['candidates']:,} short evaluations; "
                f"{data['promoted_5000']:,} long checks; "
                f"{data['long_persistent']:,} long persistent; "
                f"{data['occupied_cells']:,} occupied MAP cells; "
                f"{data['total_steps']:,} simulated steps."
            )
    if counts["cma_refinement"]:
        c = counts["cma_refinement"]
        lines.append(
            f"- Sep-CMA-ES: {c['seeds']} seeds, {c['generations']} "
            f"generations, {c['short_evaluations']:,} short and "
            f"{c['long_evaluations']:,} long evaluations, "
            f"{c['population_long_persistent']:,} long-persistent "
            f"candidate evaluations and {c['long_persistent']} "
            "long-persistent generation winners, "
            f"{c['total_steps']:,} simulated steps."
        )
    if counts.get("vast_cost_estimate"):
        cost = counts["vast_cost_estimate"]
        lines += [
            "",
            "## Experimental cost",
            "",
            f"The RTX 5090 instance was listed at ${cost['hourly_usd']:.3f}/h. "
            f"The estimated rental through report generation was "
            f"${cost['estimated_usd']:.2f} over "
            f"{cost['elapsed_hours']:.2f} hours, excluding variable "
            "network charges. Final rental cost should be read from "
            "the Vast.ai invoice after the instance is stopped.",
        ]
    lines += [
        "",
        "The random baseline uses the same 1.5 million 100-step "
        "evaluations and one 5,000-step promotion per batch of 64 as "
        "neighborhood plus QD. Short-horizon MAP cells are provisional; "
        "5,000-step promotions are listed separately. The comparison "
        "plot counts passing evaluations, not deduplicated discoveries; "
        "the visual archive and search proposals also have different "
        "selection histories. QD uses mass, area, extrapolated movement, "
        "and temporal activity descriptors; iso+line mutations follow "
        "LeniaBreeder, with a scrambled Sobol batch every eighth batch. "
        "The random baseline draws independently from the same archive "
        "seed and mutation-scale prior, without selecting parents from "
        "the evolving MAP archive.",
        "",
        "## Reproducibility",
        "",
        "See screen_config.json, search_config.json, verify_config.json, "
        "the complete parameter files, trajectory NPZs, and CSV/Parquet "
        "tables. The source archive and original ASAL checkpoints were "
        "read but not modified.",
        "",
        "## References",
        "",
        "- Hamon et al. (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12577699/",
        "- Faldor and Cully (2024): https://leniabreeder.github.io/",
        "- CARL (2026): https://developmentalsystems.org/carl/",
        "- Visualizing Lenia Parameter Space (2026): https://arxiv.org/abs/2601.01932",
        "",
        "Reference code inspected: Hamon f29a064f8bc18877269df56661a290fc29e485bb; "
        "Leniabreeder 2d7bd133548359782741598b3a4f026d69d792a9; "
        "CARL aa12ac25921aaafe83bf2cbb51bb2e9c62fc8532. "
        "The parameter-space preprint provides no transferable numerical "
        "soliton threshold for this three-channel simulator.",
        "",
    ]
    (args.output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
