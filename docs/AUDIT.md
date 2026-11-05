This is a historical audit of the original code layout. Since the 2026-11-05
refactor, active modules live under simulation/, evaluation/, search/, and
experiments/. Former lenia_search/ and scripts/ paths below refer to that
original layout. The original workflow scripts are retained in
../archive/upstream/project-history/retired-workflow-scripts.tar.gz.

ASAL source: SakanaAI/asal, commit 677ba0ea4d3b3ca78273c9906c6e84d2b1481ce7.
License: Apache-2.0, copied to ../LICENSE.

Preserved:
- substrates/lenia.py and substrates/lenia_impl.py dynamics, including
  5N7KKM, 128x128 world, 64x64 centered phenotype, 256-step default rollout,
  and dynamic parameter clip of 1.
- Supervised target loss: negative mean of aligned normalized image/text
  embedding cosine similarity.
- Sep-CMA-ES ask/tell pipeline and default sigma 0.1.

Changes:
- Only Lenia and supervised target search are included in the engine.
- Sampled-frame rollout avoids materializing all world states.
- Image encoding is batched and interchangeable; JAX/Torch DLPack transfers
  sampled frames between frameworks.
- Periodic checkpointing and resume, explicit compute budgets, and saved
  result artifacts are provided.

CPU validation: optimized sampled frames matched ASAL exactly for an 8-step,
2-sample Lenia rollout (maximum absolute error 0.0). Sep-CMA-ES ask/tell
worked with evosax 0.1.6. A two-iteration CLI smoke test with a stub encoder
saved artifacts and resumed from its checkpoint. Python compilation and Bash
syntax checks passed. MobileCLIP2-S0 and S2 architectures instantiated and
reparameterized on CPU with the pinned OpenCLIP source and timm 1.0.30.
The google/siglip2-base-patch16-224 checkpoint currently exposes a `siglip`
model and `SiglipImageProcessor`; encoder input handling follows the loaded
model type, so standard NCHW pixels are used for that checkpoint.
GPU validation (2026-11-03): The RTX 6000 Ada instance 58980056 used offer
58197217, estimated $0.603/hour including 50 GB disk. JAX 0.4.38 and
PyTorch 2.8.0+cu129 both detected the GPU. The pinned Apple MobileCLIP2-S2
checkpoint (revision 72424e7025436db18f15c3eff6ee8c7c15ad4481,
SHA-256 37c2d839a856491f2fcc82c40dc28672dbd0907235b4cd4c38dfff6457f0c09f)
produced 512-dimensional text and image embeddings. A real GPU smoke search
completed. The production run reached 1,000 generations with best cosine
0.3447445333003998. Local verification checked all 1,000 score entries,
finite parameters, eight sampled frames, and 256 decoded MP4 frames. A
checksum-based rsync dry run found no differences between local and remote
outputs. Instance 58980056 was then destroyed; Vast listed no active instances.

Illumination implementation (2026-11-04): `lenia_search/illuminate.py` adapts
`../archive/upstream/asal/main_illuminate.py` at the same source commit. It retains the
zero-vector initialization, uniform parent sampling, Gaussian mutation with
sigma 0.1, iterative removal by mean similarity to the two nearest surviving
neighbors, and the original illumination loss (mean nearest-neighbor cosine).
The Lenia substrate and 256-step final-state render are unchanged. Only the
image encoder is replaced with pinned MobileCLIP2-S2. Identical zero-vector
rollouts are evaluated once and broadcast to the initial archive. The
survival rule is accelerated with a single top-(children+2) neighbor pass;
that width is sufficient because no more than `children` members are removed
in an iteration. `scripts/check_illumination_selection.py` compared the
optimized survivor indices and loss against the upstream rule on random and
tied archives and passed on the GPU. A real 5,000-member GPU pilot completed,
then a profiled pilot reached about 0.13 seconds per generation after JIT
compilation.

The production run on instance 59037478 used an RTX 6000 Ada at an observed
rate of $0.673/hour including disk. It evaluated 320,000 mutations over
10,000 generations, retained 5,000 unique parameter vectors and embeddings,
and finished at illumination loss 0.8924117088. Local validation checked
the checkpoint, all 5,000 archive rows, 157 trajectory chunks, 24 full
256-frame videos, the UMAP atlas, and 216 remote file checksums. The instance
was destroyed and Vast showed no active instances.
