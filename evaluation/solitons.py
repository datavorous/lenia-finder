"""Numerical Lenia soliton measurements; simulation dynamics remain untouched.

References:
Hamon et al. (2025), expe/calc_categories.py in sensorimotor-lenia-search.
CARL (2026), rlca/eval/lib/runner.py in carl-release.
Faldor and Cully (2024), common.py and main_me.py in Leniabreeder.
"""

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Thresholds:
    # CARL: 0 < final mass < 10% of a single-channel board after 5,000 steps.
    max_mass_fraction: float = 0.10
    # This implementation has three channels; ignore machine-level residuals.
    min_mass_fraction: float = 1e-6
    # Hamon: last-quarter / first-quarter mass in (0, 2).
    max_phase_ratio: float = 2.0
    # Hamon: one connected object after blurring; adapted blur for R=12.
    component_activation_threshold: float = 0.05
    component_blur_radius_px: int = 6
    # Hamon: 100 px over 1,000 steps on a 256 grid; 50 px on our 128 grid.
    movement_distance_1000_px: float = 50.0
    # Calibrated extra requirements, separate from cited paper criteria.
    max_mass_cv: float = 0.35
    max_area_cv: float = 0.45
    max_radius_cv: float = 0.20
    max_occupied_fraction: float = 0.20
    min_largest_component_fraction: float = 0.90
    min_activity: float = 0.005
    max_activity_static: float = 0.02

    def to_dict(self):
        return asdict(self)


METRIC_NAMES = (
    "mass_fraction",
    "occupied_fraction",
    "com_y_px",
    "com_x_px",
    "radius_px",
    "aligned_activity",
    "border_mass_fraction",
)


def make_rollout(substrate, steps, sample_period=100):
    """Sample one untouched 128-grid trajectory without storing full movies."""
    if steps % sample_period or steps < sample_period:
        raise ValueError("steps must be a positive multiple of sample_period")
    grid = substrate.grid_size
    channels = substrate.lenia.n_channel
    R = substrate.lenia.pattern["R"] * substrate.config_lenia.world_scale
    yy, xx = jnp.mgrid[:grid, :grid]
    radius2 = (yy - grid // 2) ** 2 + (xx - grid // 2) ** 2

    def advance(state, count, params):
        def one_step(current, _):
            return substrate.step_state(None, current, params), None

        state, _ = jax.lax.scan(one_step, state, None, length=count)
        return state

    def measure(state, previous):
        world = state["carry"].world
        activity = jnp.max(world, axis=-1)
        raw_mass = jnp.sum(world)
        mass = raw_mass / (grid * grid * channels)
        area = jnp.mean(activity > 0.05)
        center = state["carry"].temp.last_center * R + state["carry"].temp.total_shift
        radius = jnp.sqrt(
            jnp.sum(world.sum(-1) * radius2) / jnp.maximum(raw_mass, 1e-12)
        )
        # The native step recenters the world, making this largely insensitive
        # to translation; locomotion is measured separately from COM.
        temporal = jnp.mean(jnp.abs(world - previous)) / jnp.maximum(mass, 1e-6)
        border = (
            jnp.sum(world[:4])
            + jnp.sum(world[-4:])
            + jnp.sum(world[:, :4])
            + jnp.sum(world[:, -4:])
        ) / jnp.maximum(raw_mass, 1e-12)
        metrics = jnp.stack(
            (mass, area, center[0], center[1], radius, temporal, border)
        )
        return metrics, world

    def one_candidate(params, key):
        state = substrate.init_state(key, params)
        previous = state["carry"].world
        state = advance(state, sample_period - 1, params)
        first, previous = measure(state, previous)

        def chunk(carry, _):
            state, previous = carry
            state = advance(state, sample_period, params)
            metrics, world = measure(state, previous)
            return (state, world), metrics

        (state, _), rest = jax.lax.scan(
            chunk, (state, previous), None, length=steps // sample_period - 1
        )
        metrics = jnp.concatenate((first[None], rest), axis=0)
        return metrics, state["carry"].world

    return jax.jit(jax.vmap(one_candidate))


def make_trial_rollout(substrate, steps=10000, mode="control", intervention_step=5000):
    """Long trajectory with an optional reproducible intervention at step 5000.

    Returns measurements every 100 steps and world snapshots every 1000.
    Parameter and initial-condition trials modify different genotype slices.
    Damage and noise trials alter the world once, during the trajectory.
    """
    if steps % 1000 or mode not in (
        "control",
        "damage",
        "noise",
        "parameter",
        "initial",
    ):
        raise ValueError("steps must be divisible by 1000; unknown mode")
    grid = substrate.grid_size
    channels = substrate.lenia.n_channel
    R = substrate.lenia.pattern["R"] * substrate.config_lenia.world_scale
    yy, xx = jnp.mgrid[:grid, :grid]
    radius2 = (yy - grid // 2) ** 2 + (xx - grid // 2) ** 2

    def one_candidate(params, key):
        if mode == "parameter":
            # The first 45 logits encode the Lenia rule. Keep the 3,072
            # initial-cell logits fixed so rule robustness is measured alone.
            perturb = jax.random.normal(key, (45,)) * 0.02
            params = params.at[:45].add(perturb)
        elif mode == "initial":
            perturb = jax.random.normal(key, params.shape) * 0.10
            params = params.at[45:].add(perturb[45:])
        state = substrate.init_state(key, params)
        previous = state["carry"].world

        def single_step(state, t):
            state = jax.lax.cond(
                t == 1,
                lambda s: s,
                lambda s: substrate.step_state(None, s, params),
                state,
            )
            if mode in ("damage", "noise"):

                def intervene(s):
                    k1, k2, k3 = jax.random.split(key, 3)
                    cy = grid // 2 + jax.random.randint(k1, (), -12, 13)
                    cx = grid // 2 + jax.random.randint(k2, (), -12, 13)
                    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
                    if mode == "damage":
                        changed = jnp.where(
                            d2[..., None] <= 12**2, 0.0, s["carry"].world
                        )
                    else:
                        noise = jax.random.normal(k3, s["carry"].world.shape) * 0.10
                        changed = jnp.clip(
                            s["carry"].world
                            + jnp.where(d2[..., None] <= 20**2, noise, 0.0),
                            0.0,
                            1.0,
                        )
                    return dict(carry=s["carry"]._replace(world=changed), img=s["img"])

                state = jax.lax.cond(
                    t == intervention_step, intervene, lambda s: s, state
                )
            return state, None

        def hundred(carry, chunk):
            state, previous = carry
            times = chunk * 100 + jnp.arange(1, 101)
            state, _ = jax.lax.scan(single_step, state, times)
            world = state["carry"].world
            raw_mass = jnp.sum(world)
            mass = raw_mass / (grid * grid * channels)
            area = jnp.mean(jnp.max(world, axis=-1) > 0.05)
            center = (
                state["carry"].temp.last_center * R + state["carry"].temp.total_shift
            )
            radius = jnp.sqrt(
                jnp.sum(world.sum(-1) * radius2) / jnp.maximum(raw_mass, 1e-12)
            )
            activity = jnp.mean(jnp.abs(world - previous)) / jnp.maximum(mass, 1e-6)
            border = (
                jnp.sum(world[:4])
                + jnp.sum(world[-4:])
                + jnp.sum(world[:, :4])
                + jnp.sum(world[:, -4:])
            ) / jnp.maximum(raw_mass, 1e-12)
            metric = jnp.stack(
                (mass, area, center[0], center[1], radius, activity, border)
            )
            return (state, world), metric

        def thousand(carry, outer):
            (state, previous), metrics = jax.lax.scan(
                hundred, carry, outer * 10 + jnp.arange(10)
            )
            return (state, previous), (metrics, state["carry"].world)

        (_, _), (metrics, snapshots) = jax.lax.scan(
            thousand, (state, previous), jnp.arange(steps // 1000)
        )
        return metrics.reshape((steps // 100, len(METRIC_NAMES))), snapshots

    return jax.jit(jax.vmap(one_candidate))


def periodic_components(world, threshold=0.05, blur_radius=6):
    """Hamon-style disk blur and labeling with periodic boundary handling."""
    activity = np.nan_to_num(np.max(world, axis=-1), nan=0.0)
    occupied = activity > threshold
    y, x = np.ogrid[-blur_radius : blur_radius + 1, -blur_radius : blur_radius + 1]
    disk = x * x + y * y <= blur_radius * blur_radius
    connected = (
        ndimage.convolve(occupied.astype(np.uint8), disk.astype(np.uint8), mode="wrap")
        > 0
    )
    labels, count = ndimage.label(connected, structure=np.ones((3, 3), dtype=np.uint8))
    h, w = occupied.shape
    parent = np.arange(count + 1)

    def root(label):
        while parent[label] != label:
            label = parent[label]
        return label

    def union(a, b):
        if a and b:
            parent[root(a)] = root(b)

    # Merge components joined across the periodic left/right and top/bottom
    # edges, including the diagonal neighbors of eight-connectivity.
    for y in range(h):
        for delta in (-1, 0, 1):
            union(labels[y, 0], labels[(y + delta) % h, -1])
    for x in range(w):
        for delta in (-1, 0, 1):
            union(labels[0, x], labels[-1, (x + delta) % w])
    component_ids = {root(int(label)) for label in labels[occupied] if label > 0}
    if component_ids:
        raw_mass = np.bincount(
            labels[occupied], weights=activity[occupied], minlength=count + 1
        )
        merged = {}
        for label in range(1, count + 1):
            merged[root(label)] = merged.get(root(label), 0.0) + raw_mass[label]
        largest = max(merged.values()) / max(float(activity[occupied].sum()), 1e-12)
    else:
        largest = 0.0
    return dict(
        component_count=int(len(component_ids)),
        largest_component_mass_fraction=largest,
        blurred_occupied_fraction=float(connected.mean()),
    )


def summarize(metrics, world, steps, thresholds=Thresholds()):
    """Return continuous metrics and independent paper/calibrated decisions."""
    a = np.asarray(metrics, dtype=np.float64)
    values = {name: a[:, i] for i, name in enumerate(METRIC_NAMES)}
    mass, area = values["mass_fraction"], values["occupied_fraction"]
    n = len(a)
    q = max(1, n // 4)
    early = float(np.nanmean(mass[:q]))
    late = float(np.nanmean(mass[-q:]))
    ratio = late / max(early, 1e-12)
    last_half_mass = mass[n // 2 :]
    last_half_area = area[n // 2 :]
    last_half_radius = values["radius_px"][n // 2 :]
    mass_cv = float(np.nanstd(last_half_mass) / max(np.nanmean(last_half_mass), 1e-12))
    area_cv = float(np.nanstd(last_half_area) / max(np.nanmean(last_half_area), 1e-12))
    radius_cv = float(
        np.nanstd(last_half_radius) / max(np.nanmean(last_half_radius), 1e-12)
    )
    center = a[:, 2:4]
    first_window = min(n, max(2, int(1000 / (steps / n))))
    first_distance = float(
        np.nanmax(np.linalg.norm(center[:first_window] - center[0], axis=1))
    )
    last_distance = float(
        np.nanmax(
            np.linalg.norm(center[-first_window:] - center[-first_window], axis=1)
        )
    )
    last_speed = last_distance / max((first_window - 1) * steps / n, 1)
    activity = float(np.nanmean(values["aligned_activity"][-q:]))
    components = periodic_components(
        world,
        thresholds.component_activation_threshold,
        thresholds.component_blur_radius_px,
    )
    finite = bool(np.isfinite(a).all())
    final_mass, final_area = float(mass[-1]), float(area[-1])
    carl_filter = bool(
        finite and 0 < final_mass < thresholds.max_mass_fraction and steps >= 5000
    )
    bounded = bool(
        finite
        and thresholds.min_mass_fraction < final_mass < thresholds.max_mass_fraction
        and final_area < thresholds.max_occupied_fraction
    )
    stable = bool(
        finite
        and 0 < ratio < thresholds.max_phase_ratio
        and mass_cv < thresholds.max_mass_cv
        and area_cv < thresholds.max_area_cv
    )
    coherent = bool(
        components["component_count"] == 1
        and components["largest_component_mass_fraction"]
        >= thresholds.min_largest_component_fraction
    )
    active = bool(activity >= thresholds.min_activity)
    moving_hamon_scaled = bool(first_distance > thresholds.movement_distance_1000_px)
    moving_sustained = bool(last_distance > thresholds.movement_distance_1000_px)
    persistent = bool(steps >= 5000 and bounded and stable and coherent)
    if not finite or final_mass <= thresholds.min_mass_fraction:
        category = "extinct"
    elif (
        final_mass >= thresholds.max_mass_fraction
        or final_area >= thresholds.max_occupied_fraction
    ):
        category = "expanding"
    elif not persistent:
        category = "transient_or_fragmented"
    elif moving_hamon_scaled and moving_sustained:
        category = "moving_soliton"
    elif active:
        category = "oscillator"
    else:
        category = "static_soliton"
    failure = (
        (mass <= thresholds.min_mass_fraction)
        | (mass >= thresholds.max_mass_fraction)
        | (area >= thresholds.max_occupied_fraction)
    )
    first_failure = int(np.flatnonzero(failure)[0]) if np.any(failure) else -1
    survival_duration = (
        steps if first_failure < 0 else int((first_failure + 1) * steps / n)
    )
    return dict(
        steps=steps,
        final_mass_fraction=final_mass,
        final_occupied_fraction=final_area,
        late_mass_mean=late,
        phase_mass_ratio=ratio,
        mass_cv=mass_cv,
        area_cv=area_cv,
        radius_cv=radius_cv,
        morphology_stable=bool(finite and radius_cv < thresholds.max_radius_cv),
        final_radius_px=float(values["radius_px"][-1]),
        aligned_activity=activity,
        first_1000_displacement_px=first_distance,
        last_1000_displacement_px=last_distance,
        last_1000_speed_px_per_step=last_speed,
        max_border_mass_fraction=float(np.nanmax(values["border_mass_fraction"])),
        first_failure_sample=first_failure,
        survival_duration_steps=survival_duration,
        finite=finite,
        carl_5000_mass_filter=carl_filter,
        bounded=bounded,
        stable=stable,
        coherent=coherent,
        active=active,
        moving_hamon_scaled=moving_hamon_scaled,
        moving_sustained=moving_sustained,
        persistent=persistent,
        category=category,
        **components,
    )
