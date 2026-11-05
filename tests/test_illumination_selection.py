"""Compare optimized archive selection with ASAL's original kill loop."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np
import torch

from search.illumination import select_archive


def original_selection(params, z, archive_size, children):
    similarity = z @ z.T
    distance = -similarity
    np.fill_diagonal(distance, np.inf)
    killed = []
    for _ in range(children):
        index = int(np.argmin(np.sort(distance, axis=1)[:, :2].mean(axis=1)))
        killed.append(index)
        distance[:, index] = np.inf
        distance[index, :] = np.inf
    kept = np.setdiff1d(np.arange(len(z)), killed, assume_unique=True)
    remaining = z[kept]
    sim = remaining @ remaining.T
    np.fill_diagonal(sim, -np.inf)
    return params[kept], remaining, float(sim.max(axis=1).mean())


def main():
    rng = np.random.default_rng(7)
    size, children, dimensions, embed_dim = 16, 4, 6, 12
    device = "cpu"
    for duplicate in (False, True):
        params = rng.normal(size=(size + children, dimensions)).astype(np.float32)
        z = rng.normal(size=(size + children, embed_dim)).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        if duplicate:
            z[:5] = z[0]
        expected_params, expected_z, expected_loss = original_selection(
            params, z, size, children
        )
        result_params, result_z, result_loss = select_archive(
            jnp.asarray(params[:size]),
            torch.as_tensor(z[:size], device=device),
            jnp.asarray(params[size:]),
            torch.as_tensor(z[size:], device=device),
            children,
        )
        np.testing.assert_allclose(
            np.asarray(result_params), expected_params, atol=1e-6
        )
        np.testing.assert_allclose(result_z.cpu().numpy(), expected_z, atol=1e-6)
        np.testing.assert_allclose(float(result_loss), expected_loss, atol=1e-6)
    print(
        "Optimized 2-neighbor kill loop matches upstream ASAL on random and tied archives"
    )


if __name__ == "__main__":
    main()
