"""Lenia update adapted from LeniaBreeder; numerical operations are kept in their original order."""

from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from .patterns import patterns

Carry = namedtuple("Carry", ["world", "param", "asset", "temp"])
Accum = namedtuple("Accum", ["phenotype", "stats"])
Param = namedtuple("Params", ["m", "s", "h"])
Asset = namedtuple("Asset", ["fK", "X", "reshape_c_k", "reshape_k_c", "R", "T"])
Temp = namedtuple("Temp", ["last_center", "last_shift", "total_shift", "last_angle"])
Stats = namedtuple(
    "Stats",
    [
        "mass",
        "center_x",
        "center_y",
        "linear_velocity",
        "angle",
        "angular_velocity",
        "is_empty",
        "is_full",
        "is_spread",
    ],
)


bell = lambda x, mean, stdev: jnp.exp(-(((x - mean) / stdev) ** 2) / 2)
growth = lambda x, mean, stdev: 2 * bell(x, mean, stdev) - 1


@dataclass
class ConfigLenia:
    pattern_id: str = "VT049W"

    world_size: int = 128
    world_scale: int = 1

    n_params_size: int = 3
    n_cells_size: int = 32


class LeniaDynamics:
    def __init__(self, config: ConfigLenia):
        self._config = config
        self.pattern = patterns[self._config.pattern_id]

        self.n_kernel = len(self.pattern["kernels"])  # k, number of kernels
        self.n_channel = len(self.pattern["cells"])  # c, number of channels
        self.n_params = (
            self._config.n_params_size * self.n_kernel
        )  # p*k, number of parameters inside genotype
        self.n_cells = (
            self._config.n_cells_size * self._config.n_cells_size * self.n_channel
        )  # e*e*c, number of embryo cells inside genotype

    def create_world_from_cells(self, cells):
        mid = self._config.world_size // 2

        scaled_cells = cells.repeat(self._config.world_scale, axis=-3).repeat(
            self._config.world_scale, axis=-2
        )
        cy, cx = scaled_cells.shape[0], scaled_cells.shape[1]

        A = jnp.zeros(
            (self._config.world_size, self._config.world_size, self.n_channel)
        )  # (y, x, c,)
        A = A.at[
            mid - cx // 2 : mid + cx - cx // 2, mid - cy // 2 : mid + cy - cy // 2, :
        ].set(scaled_cells)
        return A

    def load_pattern(self, pattern):
        cells = jnp.transpose(
            jnp.asarray(pattern["cells"]), axes=[1, 2, 0]
        )  # (y, x, c,)
        kernels = pattern["kernels"]
        R = pattern["R"] * self._config.world_scale
        T = pattern["T"]

        m = jnp.array([k["m"] for k in kernels])  # (k,)
        s = jnp.array([k["s"] for k in kernels])  # (k,)
        h = jnp.array([k["h"] for k in kernels])  # (k,)
        init_params = jnp.vstack([m, s, h])  # (p, k,)

        reshape_c_k = jnp.zeros(shape=(self.n_channel, self.n_kernel))  # (c, k,)
        reshape_k_c = jnp.zeros(shape=(self.n_kernel, self.n_channel))  # (k, c,)
        for i, k in enumerate(kernels):
            reshape_c_k = reshape_c_k.at[k["c0"], i].set(1.0)
            reshape_k_c = reshape_k_c.at[i, k["c1"]].set(1.0)

        mid = self._config.world_size // 2
        X = jnp.mgrid[-mid:mid, -mid:mid] / R  # (d, y, x,), coordinates
        D = jnp.linalg.norm(X, axis=0)  # (y, x,), distance from origin
        Ds = [D * len(k["b"]) / k["r"] for k in kernels]  # (y, x,)*k
        Ks = [
            (D < len(k["b"]))
            * jnp.asarray(k["b"])[jnp.minimum(D.astype(int), len(k["b"]) - 1)]
            * bell(D % 1, 0.5, 0.15)
            for D, k in zip(Ds, kernels)
        ]  # (x, y,)*k
        K = jnp.dstack(Ks)  # (y, x, k,), kernels
        nK = K / jnp.sum(
            K, axis=(0, 1), keepdims=True
        )  # (y, x, k,), normalized kernels
        fK = jnp.fft.fft2(
            jnp.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1)
        )  # (y, x, k,), FFT of kernels

        cy, cx = cells.shape[0], cells.shape[1]
        py, px = self._config.n_cells_size - cy, self._config.n_cells_size - cx
        init_cells = jnp.pad(
            cells,
            pad_width=((py // 2, py - py // 2), (px // 2, px - px // 2), (0, 0)),
            mode="constant",
        )  # (e, e, c,)

        A = self.create_world_from_cells(init_cells)

        init_carry = Carry(
            world=A,
            param=Param(m, s, h),
            asset=Asset(fK, X, reshape_c_k, reshape_k_c, R, T),
            temp=Temp(
                jnp.zeros(2), jnp.zeros(2, dtype=int), jnp.zeros(2, dtype=int), 0.0
            ),
        )
        init_genotype = jnp.concatenate([init_params.flatten(), init_cells.flatten()])
        return init_carry, init_genotype

    def express_genotype(self, carry, genotype):
        params = genotype[: self.n_params].reshape(
            (self._config.n_params_size, self.n_kernel)
        )
        cells = genotype[self.n_params :].reshape(
            (self._config.n_cells_size, self._config.n_cells_size, self.n_channel)
        )

        m, s, h = params
        A = self.create_world_from_cells(cells)

        carry = carry._replace(world=A)
        carry = carry._replace(param=Param(m, s, h))
        return carry

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "phenotype_size",
            "center_phenotype",
            "record_phenotype",
        ),
    )
    def step(
        self,
        carry: Carry,
        unused: Any,
        phenotype_size,
        center_phenotype,
        record_phenotype,
    ):
        A = carry.world
        m, s, h = carry.param
        fK, X, reshape_c_k, reshape_k_c, R, T = carry.asset
        last_center, last_shift, total_shift, last_angle = carry.temp
        m = m[None, None, ...]  # (1, 1, k,)
        s = s[None, None, ...]  # (1, 1, k,)
        h = h[None, None, ...]  # (1, 1, k,)

        # center world for accurate calculation of center and velocity
        A = jnp.roll(A, -last_shift, axis=(-3, -2))  # (y, x, c,)

        # Lenia step
        fA = jnp.fft.fft2(A, axes=(-3, -2))  # (y, x, c,)
        fA_k = jnp.dot(fA, reshape_c_k)  # (y, x, k,)
        U_k = jnp.real(jnp.fft.ifft2(fK * fA_k, axes=(-3, -2)))  # (y, x, k,)
        G_k = growth(U_k, m, s) * h  # (y, x, k,)
        G = jnp.dot(G_k, reshape_k_c)  # (y, x, c,)
        next_A = jnp.clip(A + 1 / T * G, 0, 1)  # (y, x, c,)

        m00 = A.sum()
        AX = next_A.sum(axis=-1)[None, ...] * X  # (d, y, x,)
        center = AX.sum(axis=(-2, -1)) / m00  # (d,)
        shift = (center * R).astype(int)
        total_shift += shift

        if record_phenotype:
            if center_phenotype:
                phenotype = next_A
            else:
                phenotype = jnp.roll(next_A, total_shift - shift, axis=(0, 1))
            mid = self._config.world_size // 2
            half_size = phenotype_size // 2
            phenotype = phenotype[
                mid - half_size : mid + half_size, mid - half_size : mid + half_size
            ]
        else:
            phenotype = None

        mass = m00 / R / R
        actual_center = center + total_shift / R
        center_diff = center - last_center + last_shift / R
        linear_velocity = jnp.linalg.norm(center_diff) * T

        angle = (
            jnp.arctan2(center_diff[1], center_diff[0]) / jnp.pi
        )  # angle = [-1.0, 1.0]
        angle_diff = (angle - last_angle + 3) % 2 - 1
        angle_diff = jax.lax.cond(
            linear_velocity > 0.01, lambda: angle_diff, lambda: 0.0
        )
        angular_velocity = angle_diff * T

        is_empty = (next_A < 0.1).all(axis=(-3, -2)).any()
        borders = (
            next_A[..., 0, :, :].sum()
            + next_A[..., -1, :, :].sum()
            + next_A[..., :, 0, :].sum()
            + next_A[..., :, -1, :].sum()
        )
        is_full = borders > 0.1
        is_spread = (
            A[
                mid - half_size : mid + half_size, mid - half_size : mid + half_size
            ].sum()
            / m00
            < 0.9
        )

        carry = carry._replace(world=next_A)
        carry = carry._replace(temp=Temp(center, shift, total_shift, angle))
        stats = Stats(
            mass,
            actual_center[1],
            -actual_center[0],
            linear_velocity,
            angle,
            angular_velocity,
            is_empty,
            is_full,
            is_spread,
        )
        accum = Accum(phenotype, stats)
        return carry, accum
