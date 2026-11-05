# Lenia design and simulation

This mini-project searches for autonomous patterns in one three-channel Lenia family. A configuration contains 45 growth-parameter offsets and a 32×32×3 initial patch. The workflow is:

**Generate configurations → Simulate → Evaluate → Select → Repeat.**

The native simulator uses a 128×128 periodic grid, 15 kernels, radius 12, and time scale 2. It keeps the Lenia update, parameter transformation, and seed used in the completed experiments. Numerical search needs no vision model. MobileCLIP2 is used only for text-guided search and visual illumination.

## Code

- simulation/: seed patterns, unchanged Lenia dynamics, genotype interface, and trajectory sampling.
- evaluation/: numerical persistence and movement measurements; optional MobileCLIP2 embeddings.
- search/: text-guided Sep-CMA-ES, visual illumination, neighborhood/MAP-Elites/random search, and numerical refinement.
- experiments/: archive screening, verification, visualization, and reports.
- tests/: small CPU regressions recorded before the refactor.
- archive/runs/: completed results, checkpoints, videos, plots, and configurations. archive/upstream/: cited source repositories and historical scripts. Both are kept locally and ignored by Git.
- runs/: output from new experiments, also ignored by Git.

The [mini-project report](MINI_PROJECT_REPORT.md) explains the methods and measured outcomes. The [numerical results report](archive/runs/soliton-work/soliton-results/REPORT.md) gives the screening thresholds and verified candidates. Report figures and saved results require the local `archive/` directory.

## Install

Use Python 3.10–3.12. For simulation and numerical search:

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

For MobileCLIP2 searches, install the optional visual dependencies. On an NVIDIA Linux machine, install the matching CUDA JAX wheel. Reports and atlas plots have a separate dependency set:

~~~bash
pip install -r requirements-visual.txt
pip install 'jax[cuda12]==0.4.38'  # NVIDIA Linux only
pip install -r requirements-report.txt  # reports and atlas only
~~~

The visual package is not imported by the simulator or numerical search. Use python -m MODULE --help for every experiment's options.

## Run the main experiments

These examples use new output directories, so the original results remain intact. The searches checkpoint and can resume from the same output directory.

**Text-guided design** with the ASAL target:

~~~bash
python -m search.target "a biological cell under the microscope" \
  --encoder mobileclip2-s2 --device cuda:0 --iterations 1000 \
  --population 16 --simulation-batch 16 --inference-batch 128 \
  --rollout-steps 256 --samples 8 --seed 0 \
  --output runs/new-cell
# Continue by adding --resume and increasing --iterations.
~~~

**Visual illumination** with ASAL's two-neighbor archive selection:

~~~bash
python -m search.illumination --archive-size 5000 --children 32 \
  --iterations 10000 --sigma 0.1 --seed 0 --rollout-steps 256 \
  --simulation-batch 32 --inference-batch 32 --samples 8 \
  --output runs/new-illumination
# Continue by adding --resume and increasing --iterations.
~~~

**Numerical screening and behavior search** use the saved 5,000-member archive:

~~~bash
ARCHIVE=archive/runs/vast-59037478/illumination-s2
python -m experiments.screen_soliton_archive "$ARCHIVE" runs/new-screen \
  --stages 5000 --batch 16
python -m experiments.verify_solitons "$ARCHIVE" \
  runs/new-screen/stage-05000.csv runs/new-verify \
  --limit 180 --steps 10000 --trials 4
python -m search.behavior "$ARCHIVE" runs/new-screen/stage-05000.csv \
  runs/new-neighborhood --phase neighborhood --evaluations 500000
python -m search.behavior "$ARCHIVE" runs/new-screen/stage-05000.csv \
  runs/new-qd --phase qd --evaluations 1000000 \
  --bootstrap-state runs/new-neighborhood/state.npz
python -m search.behavior "$ARCHIVE" runs/new-screen/stage-05000.csv \
  runs/new-random --phase random --evaluations 1500000
python -m search.refine "$ARCHIVE" runs/new-screen/stage-05000.csv \
  runs/new-refinement
python -m experiments.report_solitons runs/new-screen/stage-05000.csv \
  runs/new-verify runs/new-report \
  --neighborhood runs/new-neighborhood --qd runs/new-qd --random runs/new-random
python -m experiments.render_soliton_gallery runs/new-report/catalog.csv \
  "$ARCHIVE" runs/new-gallery --limit 10
~~~

These commands launch substantial searches; reduce --max-configs, --evaluations, or --generations for a small run. Numerical outputs resume from their output directories. The atlas command is python -m experiments.plot_illumination_atlas runs/new-illumination. For high-resolution archive replays, use python -m experiments.render_diverse_illumination --help.

## Verify the refactor

On a CPU installation with the optional visual dependencies needed for the small archive-selection check:

~~~bash
python -m tests.test_regression
python -m tests.test_soliton_metrics
python -m tests.test_illumination_selection
~~~

The regression fixture compares actual native-grid trajectories, sampled images, numerical evaluator outputs, deterministic proposals, and archive selection against outputs captured before the refactor. It also loads the saved target and illumination checkpoints. No long search is run.

The pinned MobileCLIP2-S2 model was also checked on CPU against the saved cell-target frames: recomputed loss -0.344661 versus recorded loss -0.344745. The small regression suite uses the saved embeddings and scores without downloading the model.

## Attribution and data

The Lenia transition and seed patterns come from [LeniaBreeder](https://leniabreeder.github.io/), via the [ASAL repository](https://github.com/SakanaAI/asal). The text-guided and illumination strategies adapt ASAL; the numerical measurements draw on Hamon et al. and CARL, with implementation-specific thresholds documented in the results report. MobileCLIP2-S2 is Apple's model. Original source checkouts remain in archive/upstream/; retired one-time workflow scripts are retained in archive/upstream/project-history/retired-workflow-scripts.tar.gz. All completed experiment artifacts remain in archive/runs/.
