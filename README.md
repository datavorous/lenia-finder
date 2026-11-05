# Lenia design and simulation

This project searches for patterns in a Lenia cellular automaton with three channels. It generates configurations. It simulates each one and scores its behavior. The scores guide the next search step. The simulator and mathematical search work without an image model. Text search and visual diversity search use MobileCLIP2-S2.

The code has four parts:

- `simulation/` runs Lenia.
- `evaluation/` measures trajectories and image similarity.
- `search/` proposes configurations.
- `experiments/` screens archives and makes reports.

Run the commands below from the repository root.

## Install

Use Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `requirements-visual.txt` for visual searches. Install `requirements-report.txt` for plots and reports. On NVIDIA Linux, install a CUDA build of JAX that matches your driver. The recorded runs used `jax[cuda12]==0.4.38`.

## Run

Search for the original text target:

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

Screen the saved archive of 5,000 configurations. Then search around its candidates:

```bash
ARCHIVE=archive/runs/vast-59037478/illumination-s2
python -m experiments.screen_soliton_archive "$ARCHIVE" runs/screen --stages 500,1000,5000
python -m search.behavior "$ARCHIVE" runs/screen/stage-05000.csv \
  runs/neighborhood --phase neighborhood --evaluations 500000
```

`search.behavior` also accepts `--phase qd` and `--phase random`. Use `python -m <module> --help` for other options. Searches save checkpoints in their output directories. Add `--resume` to continue a visual search. These examples require substantial GPU time.

## Results and provenance

The local `archive/` directory contains completed runs, checkpoints, videos, and source snapshots. Git ignores this directory. Git also ignores new results in `runs/`. The [project report](docs/MINI_PROJECT_REPORT.md) summarizes the experiments. The [audit](docs/AUDIT.md) records the source history. The [numerical results report](archive/runs/soliton-work/soliton-results/REPORT.md) lists screening thresholds and verified candidates. The report figures require the local archive.

Lenia dynamics and seed patterns come from [LeniaBreeder](https://leniabreeder.github.io/) through [ASAL](https://github.com/SakanaAI/asal). The visual search methods adapt ASAL. Numerical screening draws on Hamon et al. and CARL. The original source checkouts are in `archive/upstream/`.

## Check

```bash
python -m tests.test_regression
python -m tests.test_soliton_metrics
python -m tests.test_illumination_selection
```

The regression test uses saved checkpoints and a CPU reference fixture. It does not start a research run.
