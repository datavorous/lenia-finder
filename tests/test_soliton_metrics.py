"""Independent detector checks for periodic components and motion classification."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

from evaluation.solitons import periodic_components, summarize

world = np.zeros((128, 128, 3), np.float32)
world[60:68, :3, 0] = 0.8
world[60:68, -3:, 0] = 0.8
assert periodic_components(world)["component_count"] == 1
world[60:68, 60:68, 0] = 0.8
assert periodic_components(world)["component_count"] == 2
world[60:68, 60:68, 0] = 0

metrics = np.zeros((50, 7), np.float32)
metrics[:, 0] = 0.02
metrics[:, 1] = 0.03
metrics[:, 4] = 12
metrics[:, 5] = 0.10
moving = metrics.copy()
moving[:, 2] = np.arange(50) * 10
assert summarize(moving, world, 5000)["category"] == "moving_soliton"
assert summarize(metrics, world, 5000)["category"] == "oscillator"
metrics[:, 5] = 0
assert summarize(metrics, world, 5000)["category"] == "static_soliton"
metrics[:, 0] = 0
assert summarize(metrics, np.zeros_like(world), 5000)["category"] == "extinct"
print("Periodic components and static/oscillatory/moving/extinct decisions passed")
