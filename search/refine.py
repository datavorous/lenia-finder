#!/usr/bin/env python3
"""Diagonal (separable) CMA-ES refinement of persistent moving Lenia seeds."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from evaluation.solitons import make_rollout, summarize
from simulation.lenia import Lenia


def append(path, row):
    first = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if first:
            w.writeheader()
        w.writerow(row)


def score(series):
    m = np.asarray(series)
    mass = m[:, -1, 0]
    area = m[:, -1, 1]
    recent = m[:, m.shape[1] // 2 :, 0]
    cv = np.std(recent, axis=1) / np.maximum(np.mean(recent, axis=1), 1e-9)
    speed = np.linalg.norm(m[:, -1, 2:4] - m[:, 0, 2:4], axis=1)
    bounded = (
        np.isfinite(m).all(axis=(1, 2)) & (mass > 1e-6) & (mass < 0.10) & (area < 0.20)
    )
    quality = np.where(
        bounded, 1.0 / (1.0 + 10.0 * cv) + 0.20 * np.minimum(speed / 50.0, 1.0), -1.0
    )
    return quality, speed, cv


def run_one(seed_index, params, output, args, quick, long):
    target = output / f"archive-{seed_index:04d}"
    target.mkdir(exist_ok=True)
    checkpoint = target / "cma_state.npz"
    dim = len(params)
    population = args.population
    mu = population // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    # Standard sep-CMA-ES path and diagonal covariance update constants.
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    # Separable CMA uses dimension-amplified covariance learning rates.
    c1 = min(1.0, c1 * (dim + 2) / 3)
    cmu = min(1 - c1, cmu * (dim + 2) / 3)
    ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            generation = int(z["generation"])
            mean, diag, ps, pc = (
                z[k].astype(np.float64) for k in ("mean", "diag", "ps", "pc")
            )
            sigma = float(z["sigma"])
            best = z["best"]
            best_score = float(z["best_score"])
    else:
        generation = 0
        mean = params.astype(np.float64)
        diag = np.ones(dim, np.float64)
        ps = np.zeros(dim, np.float64)
        pc = np.zeros(dim, np.float64)
        sigma = args.sigma
        best = params.copy()
        best_score = -np.inf
    for g in range(generation, args.generations):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, seed_index, g]))
        z = rng.standard_normal((population, dim))
        y = z * np.sqrt(diag)
        candidates = (mean[None] + sigma * y).astype(np.float32)
        candidates[:, :45] = np.clip(candidates[:, :45], -1.0, 1.0)
        keys = jnp.stack(
            [
                jax.random.fold_in(
                    jax.random.PRNGKey(args.seed),
                    seed_index * 100000 + g * population + i,
                )
                for i in range(population)
            ]
        )
        metrics, _ = quick(jnp.asarray(candidates), keys)
        short_metrics = np.asarray(jax.device_get(metrics))
        short_quality, speed, cv = score(short_metrics)
        long_metrics, long_world = long(jnp.asarray(candidates), keys)
        long_metrics = np.asarray(jax.device_get(long_metrics))
        long_world = np.asarray(jax.device_get(long_world))
        summaries = [
            summarize(long_metrics[i], long_world[i], 5000) for i in range(population)
        ]
        quality = np.asarray(
            [
                (
                    1.0 / (1.0 + 5.0 * s["mass_cv"])
                    + min(s["last_1000_displacement_px"] / 50.0, 1.0)
                )
                if s["persistent"]
                and s["max_border_mass_fraction"] < 0.10
                and s["final_radius_px"] < 40.0
                and s["radius_cv"] < 0.20
                else -1.0
                for s in summaries
            ],
            np.float64,
        )
        order = np.argsort(quality)[::-1]
        parents = order[:mu]
        ymean = np.sum(weights[:, None] * y[parents], axis=0)
        zmean = ymean / np.sqrt(diag)
        old_sigma = sigma
        mean += old_sigma * ymean
        mean[:45] = np.clip(mean[:45], -1.0, 1.0)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
        norm_ps = np.linalg.norm(ps)
        sigma = float(
            np.clip(sigma * np.exp(cs / ds * (norm_ps / chi - 1)), 0.001, 1.0)
        )
        hsig = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chi < 1.4 + 2 / (
            dim + 1
        )
        pc = (1 - cc) * pc + (int(hsig) * np.sqrt(cc * (2 - cc) * mueff) * ymean)
        diag = (
            (1 - c1 - cmu) * diag
            + c1 * (pc**2 + (1 - int(hsig)) * cc * (2 - cc) * diag)
            + cmu * np.sum(weights[:, None] * y[parents] ** 2, axis=0)
        )
        diag = np.clip(diag, 1e-6, 1e6)
        winner = int(order[0])
        if quality[winner] > best_score:
            best_score = float(quality[winner])
            best = candidates[winner].copy()
        verified = summaries[winner]
        np.savez_compressed(
            target / f"generation-{g:04d}.npz",
            parameters=candidates[winner],
            candidate_parameters=candidates,
            short_quality=short_quality,
            long_quality=quality,
            short_metrics=short_metrics,
            long_metrics=long_metrics,
        )
        append(
            target / "generations.csv",
            dict(
                generation=g,
                short_quality=float(short_quality[winner]),
                short_displacement=float(speed[winner]),
                short_mass_cv=float(cv[winner]),
                sigma=sigma,
                long_persistent=verified["persistent"],
                long_category=verified["category"],
                long_mass=verified["final_mass_fraction"],
                long_displacement=verified["last_1000_displacement_px"],
                long_quality=float(quality[winner]),
                population_persistent=sum(s["persistent"] for s in summaries),
                population_moving=sum(
                    s["category"] == "moving_soliton" for s in summaries
                ),
            ),
        )
        temp = target / "cma_state.tmp.npz"
        np.savez_compressed(
            temp,
            generation=g + 1,
            mean=mean,
            diag=diag,
            ps=ps,
            pc=pc,
            sigma=sigma,
            best=best,
            best_score=best_score,
        )
        temp.replace(checkpoint)
        print(
            f"seed {seed_index} generation {g + 1}/{args.generations}: "
            f"score={quality[winner]:.3f} long={verified['category']} "
            f"persistent={sum(s['persistent'] for s in summaries)}/{population}",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("screen_csv", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=8288)
    args = p.parse_args()
    if jax.default_backend() != "gpu":
        p.error("GPU backend required")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    config.update(
        short_steps=500,
        long_steps=5000,
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
    )
    path = args.output / "refine_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        p.error("Cannot resume with changed configuration")
    path.write_text(json.dumps(config, indent=2) + "\n")
    params = np.load(args.archive / "parameters.npy", mmap_mode="r")
    with args.screen_csv.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] == "moving_soliton"]
    rows.sort(
        key=lambda r: (-float(r["last_1000_displacement_px"]), float(r["mass_cv"]))
    )
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    quick = make_rollout(substrate, 500, 100)
    long = make_rollout(substrate, 5000, 100)
    for r in rows[: args.seeds]:
        index = int(r["archive_index"])
        run_one(index, np.asarray(params[index]), args.output, args, quick, long)


if __name__ == "__main__":
    main()
