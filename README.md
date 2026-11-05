# Lenia design and simulation

This project searches for patterns in a three-channel Lenia cellular automaton. It generates configurations, simulates them, scores the resulting behavior, and uses the scores to choose the next configurations. The simulator and mathematical search run without an image model. Text-guided and visual diversity searches use MobileCLIP2-S2.

The code is split by task: `simulation/` runs Lenia, `evaluation/` measures trajectories and image similarity, `search/` proposes configurations, and `experiments/` screens archives and makes reports. All commands below run from the repository root.

## Install

Use Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For visual searches, install `requirements-visual.txt`. For plots and reports, install `requirements-report.txt`. On NVIDIA Linux, install the CUDA build of JAX that matches your driver; the recorded runs used `jax[cuda12]==0.4.38`.

## Run

Text-guided search for the original target:

```bash
python -m search.target "a biological cell under the microscope" \
  --encoder mobileclip2-s2 --device cuda:0 --iterations 1000 \
  --population 16 --simulation-batch 16 --inference-batch 128 \
  --output runs/cell
```

Visual diversity search:

```bash
python -m search.illumination --archive-size 5000 --children 32 \
  --iterations 10000 --output runs/illumination
```

Screen the saved 5,000-configuration archive, then search around its candidates:

```bash
ARCHIVE=archive/runs/vast-59037478/illumination-s2
python -m experiments.screen_soliton_archive "$ARCHIVE" runs/screen --stages 500,1000,5000
python -m search.behavior "$ARCHIVE" runs/screen/stage-05000.csv \
  runs/neighborhood --phase neighborhood --evaluations 500000
```

`search.behavior` also accepts `--phase qd` and `--phase random`. Use `python -m <module> --help` for other options. Searches write checkpoints to their output directories; add `--resume` to continue a visual search there. These full-size examples require substantial GPU time.

## Results and provenance

Completed runs, checkpoints, videos, and source snapshots are in the local `archive/` directory. It is ignored by Git; new results in `runs/` are ignored too. The [mini-project report](docs/MINI_PROJECT_REPORT.md) summarizes the experiments, and the [audit](docs/AUDIT.md) records the source history. The [numerical results report](archive/runs/soliton-work/soliton-results/REPORT.md) contains screening thresholds and verified candidates. The reports' figures require the local archive.

Lenia dynamics and seed patterns come from [LeniaBreeder](https://leniabreeder.github.io/) through [ASAL](https://github.com/SakanaAI/asal). The visual search methods adapt ASAL; numerical screening draws on Hamon et al. and CARL. The original source checkouts are preserved in `archive/upstream/`.

## Check

```bash
python -m tests.test_regression
python -m tests.test_soliton_metrics
python -m tests.test_illumination_selection
```

The regression test uses saved checkpoints and the CPU reference fixture; it does not start a research run.
