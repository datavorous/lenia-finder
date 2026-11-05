#!/usr/bin/env python3
"""Resumable, batched Lenia neighborhood, random and MAP-Elites search.

The simulator is imported unchanged. All candidate generation is deterministic
from (run seed, phase, batch number); checkpoints contain the evolving archive.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from evaluation.solitons import make_rollout, summarize
from simulation.lenia import Lenia


def rows_from_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pick_seeds(parameters, screening, limit=512):
    rows = rows_from_csv(screening)
    candidates = [r for r in rows if r["bounded"] == "True" and r["coherent"] == "True"]
    candidates.sort(
        key=lambda r: (
            r["persistent"] == "True",
            -float(r["mass_cv"]),
            -float(r["area_cv"]),
        ),
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            rows,
            key=lambda r: (r["bounded"] == "True", -float(r["mass_cv"])),
            reverse=True,
        )
    # Preserve movement and oscillator seeds even when their variability makes
    # them rank below nearly static configurations on mass CV.
    selected = []
    used = set()
    for category in ("moving_soliton", "oscillator", "static_soliton"):
        for row in candidates:
            if row["category"] == category and len(selected) < limit:
                selected.append(row)
                used.add(int(row["archive_index"]))
    for row in candidates:
        index = int(row["archive_index"])
        if len(selected) >= limit:
            break
        if index not in used:
            selected.append(row)
            used.add(index)
    return np.asarray(
        parameters[[int(r["archive_index"]) for r in selected]], dtype=np.float32
    ), selected


def quick_metrics(metrics, sample_period):
    m = np.asarray(metrics)
    mass = m[:, -1, 0]
    area = m[:, -1, 1]
    activity = m[:, -1, 5]
    disp = np.linalg.norm(m[:, -1, 2:4] - m[:, 0, 2:4], axis=-1)
    n = m.shape[1]
    disp_1000 = disp * 1000.0 / max((n - 1) * sample_period, 1)
    recent = m[:, n // 2 :, 0]
    cv = np.std(recent, axis=1) / np.maximum(np.mean(recent, axis=1), 1e-9)
    bounded = (
        np.isfinite(m).all(axis=(1, 2)) & (mass > 1e-6) & (mass < 0.10) & (area < 0.20)
    )
    # A continuous ranking score; later persistence and coherence checks are
    # separate and never inferred from this short rollout.
    quality = np.where(
        bounded,
        1.0 - np.minimum(cv, 2.0) / 2.0 + np.minimum(disp_1000 / 50.0, 1.0) * 0.1,
        -1.0,
    )
    descriptor = np.stack((mass, area, disp_1000 / 100.0, activity), axis=1)
    return dict(
        mass=mass,
        area=area,
        activity=activity,
        displacement=disp,
        displacement_1000_extrapolated=disp_1000,
        mass_cv=cv,
        bounded=bounded,
        quality=quality,
        descriptor=descriptor,
    )


def cells(descriptors):
    scales = np.asarray([0.1, 0.2, 1.0, 2.0], dtype=np.float32)
    scaled = np.nan_to_num(descriptors / scales * 8, nan=0.0, posinf=7.0, neginf=0.0)
    return np.clip(scaled, 0, 7).astype(np.int16)


def candidate_batch(phase, seeds, elites, batch_number, size, seed):
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [seed, batch_number, {"neighborhood": 1, "qd": 2, "random": 3}[phase]]
        )
    )
    if phase == "random":
        # Same archived-prior center and mutation-scale mixture as neighborhood
        # search; no fitness-based parent selection.
        centers = seeds[rng.integers(len(seeds), size=size)]
    elif phase == "qd" and len(elites):
        centers = elites[rng.integers(len(elites), size=size)]
    else:
        centers = seeds[rng.integers(len(seeds), size=size)]
    scales = np.asarray([0.005, 0.02, 0.08, 0.2, 0.5, 1.0], dtype=np.float32)
    sigma = scales[rng.integers(len(scales), size=size)]
    if phase == "qd" and batch_number % 8 == 0:
        # Scrambled Sobol draws provide quasi-random coverage of the full
        # 3,117-dimensional mutation space without changing the prior scale.
        sampler = qmc.Sobol(
            d=centers.shape[1], scramble=True, seed=int(rng.integers(2**31))
        )
        uniforms = sampler.random_base2(int(np.ceil(np.log2(size))))[:size]
        noise = ndtri(np.clip(uniforms, 1e-7, 1 - 1e-7)).astype(np.float32)
    else:
        noise = rng.standard_normal((size, centers.shape[1]), dtype=np.float32)
    candidates = centers + sigma[:, None] * noise
    if phase == "qd" and len(elites) > 1:
        # Faldor/Cully MAP-Elites iso+line variation, with scale mixture to
        # traverse both local and more distant neighborhoods.
        second = elites[rng.integers(len(elites), size=size)]
        line = rng.normal(0, 0.05, (size, 1)).astype(np.float32)
        candidates += line * (second - centers)
    # Dynamic logits are clipped by the original substrate at +/-1. Clipping
    # here only avoids storing equivalent out-of-range proposals.
    candidates[:, :45] = np.clip(candidates[:, :45], -1, 1)
    return candidates.astype(np.float32), sigma


def save_state(output, state):
    temp = output / "state.tmp.npz"
    np.savez_compressed(temp, **state)
    temp.replace(output / "state.npz")


def append_rows(path, rows):
    first = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if first:
            writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("screen_csv", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--phase", choices=("neighborhood", "qd", "random"), required=True)
    p.add_argument("--evaluations", type=int, required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--sample-period", type=int, default=20)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--promote-fraction", type=float, default=0.01)
    p.add_argument(
        "--bootstrap-state",
        type=Path,
        help="Import provisional elite cells from a completed search",
    )
    args = p.parse_args()
    if jax.default_backend() != "gpu":
        p.error("GPU JAX backend required")
    if (
        args.steps % args.sample_period
        or args.steps <= args.sample_period
        or args.evaluations < 1
    ):
        p.error(
            "steps must contain at least two samples and evaluations must be positive"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.archive / "parameters.npy", mmap_mode="r")
    seeds, seed_rows = pick_seeds(archive, args.screen_csv)
    if not len(seeds):
        p.error("No screening seeds")
    config = vars(args).copy()
    config.update(
        archive=str(args.archive.resolve()),
        screen_csv=str(args.screen_csv.resolve()),
        seed_archive_indices=[int(r["archive_index"]) for r in seed_rows],
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        dynamics="unchanged simulation.lenia.Lenia",
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    conf_path = args.output / "search_config.json"
    if conf_path.exists():
        old = json.loads(conf_path.read_text())
        for key in (
            "phase",
            "evaluations",
            "batch",
            "steps",
            "sample_period",
            "seed",
            "seed_archive_indices",
        ):
            if old[key] != config[key]:
                p.error(f"Cannot resume with changed {key}")
    else:
        conf_path.write_text(json.dumps(config, indent=2) + "\n")
    state_path = args.output / "state.npz"
    if state_path.exists():
        with np.load(state_path) as z:
            completed = int(z["completed"])
            elite_params = z["elite_params"]
            elite_quality = z["elite_quality"]
            elite_cells = z["elite_cells"]
    else:
        completed = 0
        elite_params = np.empty((0, archive.shape[1]), np.float32)
        elite_quality = np.empty(0, np.float32)
        elite_cells = np.empty((0, 4), np.int16)
        if args.bootstrap_state:
            with np.load(args.bootstrap_state) as z:
                elite_params = z["elite_params"]
                elite_quality = z["elite_quality"]
                elite_cells = z["elite_cells"]
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    rollout = make_rollout(substrate, args.steps, sample_period=args.sample_period)
    long_rollout = make_rollout(substrate, 5000, sample_period=100)
    (args.output / "candidate_chunks").mkdir(exist_ok=True)
    (args.output / "promotion_chunks").mkdir(exist_ok=True)
    start = time.monotonic()
    while completed < args.evaluations:
        batch_number = completed // args.batch
        count = min(args.batch, args.evaluations - completed)
        proposals, sigmas = candidate_batch(
            args.phase, seeds, elite_params, batch_number, count, args.seed
        )
        keys = jnp.stack(
            [
                jax.random.fold_in(jax.random.PRNGKey(args.seed), completed + i)
                for i in range(count)
            ]
        )
        series, _ = rollout(jnp.asarray(proposals), keys)
        qm = quick_metrics(jax.device_get(series), args.sample_period)
        desc_cells = cells(qm["descriptor"])
        rows = []
        for i in range(count):
            rows.append(
                dict(
                    candidate_id=completed + i,
                    sigma=float(sigmas[i]),
                    steps=args.steps,
                    mass=float(qm["mass"][i]),
                    area=float(qm["area"][i]),
                    activity=float(qm["activity"][i]),
                    displacement=float(qm["displacement"][i]),
                    displacement_1000_extrapolated=float(
                        qm["displacement_1000_extrapolated"][i]
                    ),
                    mass_cv=float(qm["mass_cv"][i]),
                    bounded=bool(qm["bounded"][i]),
                    quality=float(qm["quality"][i]),
                    cell="-".join(map(str, desc_cells[i])),
                )
            )
        chunk_path = args.output / "candidate_chunks" / f"batch-{batch_number:07d}.csv"
        if chunk_path.exists():
            chunk_path.unlink()
        append_rows(chunk_path, rows)
        # Archive updates use short-horizon descriptors and are explicitly
        # provisional. Full 5,000-step checks are saved independently.
        lookup = {tuple(c): j for j, c in enumerate(elite_cells)}
        order = np.argsort(qm["quality"])[::-1]
        cap = max(1, int(np.ceil(count * args.promote_fraction)))
        # Fixed count for every batch keeps the random baseline on the same
        # short/long simulation-step budget as each guided phase.
        promoted = [int(i) for i in order[:cap]]
        for i in order:
            if not qm["bounded"][i]:
                break
            key = tuple(desc_cells[i])
            if key not in lookup or qm["quality"][i] > elite_quality[lookup[key]]:
                if key in lookup:
                    j = lookup[key]
                    elite_params[j] = proposals[i]
                    elite_quality[j] = qm["quality"][i]
                else:
                    lookup[key] = len(elite_params)
                    elite_params = np.concatenate((elite_params, proposals[i : i + 1]))
                    elite_quality = np.append(elite_quality, qm["quality"][i])
                    elite_cells = np.concatenate((elite_cells, desc_cells[i : i + 1]))
        if promoted:
            long_keys = keys[np.asarray(promoted)]
            long_metrics, long_worlds = long_rollout(
                jnp.asarray(proposals[promoted]), long_keys
            )
            long_metrics, long_worlds = map(
                lambda x: np.asarray(jax.device_get(x)), (long_metrics, long_worlds)
            )
            verified = []
            for j, i in enumerate(promoted):
                result = summarize(long_metrics[j], long_worlds[j], 5000)
                verified.append(dict(candidate_id=completed + i, **result))
            promoted_path = (
                args.output / "promotion_chunks" / f"batch-{batch_number:07d}.csv"
            )
            if promoted_path.exists():
                promoted_path.unlink()
            append_rows(promoted_path, verified)
            np.savez_compressed(
                args.output / f"promoted-{batch_number:07d}.npz",
                candidate_ids=completed + np.asarray(promoted),
                parameters=proposals[promoted],
                metrics=long_metrics,
            )
        completed += count
        if batch_number % 100 == 0 or completed == args.evaluations:
            save_state(
                args.output,
                dict(
                    completed=completed,
                    elite_params=elite_params,
                    elite_quality=elite_quality,
                    elite_cells=elite_cells,
                ),
            )
        if batch_number % 10 == 0 or completed == args.evaluations:
            print(
                f"{args.phase}: {completed}/{args.evaluations}; "
                f"cells {len(elite_params)}; promoted {len(promoted)}; "
                f"{time.monotonic() - start:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
