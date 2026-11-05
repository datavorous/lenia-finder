"""Small CPU regressions recorded before the code was reorganized."""

import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import torch
import evosax

from evaluation.embeddings import target_loss
from evaluation.solitons import summarize
from search.behavior import candidate_batch, cells, quick_metrics
from search.illumination import prepare_functions, select_archive
from simulation.lenia import Lenia
from simulation.trajectory import sampled_frames

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = np.load(Path(__file__).with_name("reference.npz"))


def same(name, value):
    np.testing.assert_allclose(np.asarray(value), EXPECTED[name], rtol=1e-6, atol=1e-6)


def main():
    params = np.load(ROOT / "archive/runs/vast-58980056/cell-s2/best.npz")["params"]
    substrate = Lenia(
        grid_size=128,
        center_phenotype=True,
        phenotype_size=64,
        start_pattern="5N7KKM",
        clip1=1.0,
    )
    key = jax.random.PRNGKey(0)
    state = substrate.init_state(key, jnp.asarray(params))
    worlds = [np.asarray(state["carry"].world)]
    for _ in range(3):
        state = substrate.step_state(key, state, jnp.asarray(params))
        worlds.append(np.asarray(state["carry"].world))
    same("worlds", worlds)
    same("frames", sampled_frames(substrate, key, jnp.asarray(params), 8, 2))

    rng = np.random.default_rng(11)
    population = rng.normal(size=(16, 6)).astype(np.float32)
    features = rng.normal(size=(16, 12)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    selected, embeddings, loss = select_archive(
        jnp.asarray(population[:12]),
        torch.from_numpy(features[:12]),
        jnp.asarray(population[12:]),
        torch.from_numpy(features[12:]),
        4,
    )
    same("selected_params", selected)
    same("selected_z", embeddings.numpy())
    same("selected_loss", loss)

    propose, *_ = prepare_functions(
        None,
        SimpleNamespace(
            archive_size=5, children=2, sigma=0.1, rollout_steps=8, samples=2
        ),
        6,
    )
    children, keys = propose(jnp.zeros((5, 6)), jax.random.PRNGKey(3))
    np.testing.assert_allclose(
        np.asarray(children)[0],
        [-0.15303713, 0.12958914, 0.1931541, -0.07921397, -0.24266869, 0.01462674],
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(keys), [[2461983011, 2982780331], [675202576, 137302448]]
    )

    text = torch.eye(2)
    aligned = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    swapped = aligned.flip(-1)
    torch.testing.assert_close(
        target_loss(torch.cat((aligned, swapped)), text), torch.tensor([-1.0, 0.0])
    )

    seeds = np.zeros((3, len(params)), dtype=np.float32)
    seeds[1, :45] = 0.1
    seeds[2, 45:] = 0.02
    proposals, scales = candidate_batch(
        "neighborhood", seeds, np.empty((0, len(params)), dtype=np.float32), 3, 4, 7
    )
    same("candidates", proposals)
    same("sigma", scales)

    strategy = evosax.Sep_CMA_ES(popsize=4, num_dims=6, sigma_init=0.1)
    key = jax.random.PRNGKey(7)
    key, initialization_key = jax.random.split(key)
    state = strategy.initialize(initialization_key, strategy.default_params)
    key, proposal_key, _ = jax.random.split(key, 3)
    population, next_state = strategy.ask(proposal_key, state, strategy.default_params)
    state = strategy.tell(
        population, jnp.sum(population**2, axis=1), next_state, strategy.default_params
    )
    np.testing.assert_allclose(
        np.asarray(population[0]),
        [0.16183268, 0.08973837, -0.12524424, -0.02371719, -0.07892685, 0.09077752],
        atol=1e-7,
    )
    np.testing.assert_allclose(state.best_fitness, 0.0565936453640461, atol=1e-7)

    metrics = np.zeros((2, 5, 7), dtype=np.float32)
    metrics[:, :, 0] = 0.01
    metrics[:, :, 1] = 0.05
    metrics[:, :, 5] = 0.01
    scores = quick_metrics(metrics, 100)
    same("descriptor", scores["descriptor"])
    same("quality", scores["quality"])
    same("cells", cells(scores["descriptor"]))
    measured = summarize(metrics[0], worlds[-1], 500)
    assert measured["category"] == "transient_or_fragmented"
    np.testing.assert_allclose(measured["final_mass_fraction"], 0.01, atol=1e-8)

    best = np.load(ROOT / "archive/runs/vast-58980056/cell-s2/best.npz")
    history = json.loads((ROOT / "archive/runs/vast-58980056/cell-s2/scores.json").read_text())
    np.testing.assert_allclose(best["loss"], history[-1]["best_loss"], atol=1e-8)
    archive = ROOT / "archive/runs/vast-59037478/illumination-s2"
    image_features = np.load(archive / "embeddings.npy", mmap_mode="r")
    nearest = np.load(archive / "nearest_cosine.npy")
    novelty = np.load(archive / "novelty.npy")
    np.testing.assert_allclose(novelty, 1 - nearest[:, 0], atol=1e-6)
    for index in (0, 17, 4686, 4999):
        similarities = np.einsum("ij,j->i", image_features, image_features[index])
        similarities[index] = -np.inf
        np.testing.assert_allclose(
            np.sort(similarities)[-2:][::-1], nearest[index], atol=1e-5
        )

    with (ROOT / "archive/runs/vast-58980056/cell-s2/checkpoint.pkl").open("rb") as handle:
        target = pickle.load(handle)
    with (ROOT / "archive/runs/vast-59037478/illumination-s2/checkpoint.pkl").open(
        "rb"
    ) as handle:
        illumination = pickle.load(handle)
    assert target["iteration"] == 1000
    assert illumination["iteration"] == 10000
    assert illumination["params"].shape == (5000, len(params))
    print("Trajectories, evaluators, seeded search, and saved checkpoints passed")


if __name__ == "__main__":
    main()
