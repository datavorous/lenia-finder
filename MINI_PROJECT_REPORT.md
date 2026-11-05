# Design and Simulation of Zero-Player Games

*Third-semester mini-project · Three credits*

**Abstract.** A zero-player game evolves according to its rules after an initial configuration is chosen. This project built an automated workflow for designing and simulating such games in Lenia, a continuous-state cellular automaton. An ASAL-derived simulator searches over growth parameters and initial cells. MobileCLIP2-S2 embeddings guide one search toward a text description and a second search toward visually diverse outcomes. Numerical measurements then screen the resulting archive for persistent, localized, active patterns. The saved experiments contain a 5,000-configuration visual archive and seven candidates that passed a stricter robustness test, including two classified as moving patterns. These are computational observations within one Lenia family, not evidence of biological life or previously unknown species.

## 1. Zero-Player Games

A **zero-player game** is a system whose state changes under fixed rules without further decisions by a player. A person may choose its rules and starting state, but the subsequent sequence of states is autonomous. Cellular automata provide a simple example: a grid of cells is updated repeatedly according to local interactions. In Conway's *Game of Life*, each cell is alive or dead. A live cell survives with two or three live neighbors, while a dead cell becomes alive with exactly three; all other cases become or remain dead [1]. Small initial patterns can disappear, persist, oscillate, or move across the grid.

Lenia extends this idea to continuous cell states and smooth, spatially distributed interactions [2]. In our implementation, each cell has three channel values in \([0,1]\). Fifteen kernels collect nearby values from specified input channels. For kernel \(k\), let \(K_k\) denote its spatial weights, \(c_0(k)\) its input channel, \(c_1(k)\) its output channel, and \(A_c(x,t)\) the state of channel \(c\) at position \(x\) and step \(t\). The update is

\[
U_k(x,t)=(K_k*A_{c_0(k)})(x,t), \qquad
g_k(U)=h_k\left[2\exp\left(-\frac{(U-m_k)^2}{2s_k^2}\right)-1\right],
\]

\[
A_c(x,t+1)=\operatorname{clip}_{[0,1]}\left(A_c(x,t)+\frac{1}{T}
\sum_{k:c_1(k)=c}g_k\bigl(U_k(x,t)\bigr)\right).
\]

Here, \(m_k\), \(s_k\), and \(h_k\) control each growth response; \(T\) sets the time scale. Convolution uses periodic boundaries, so the grid wraps at its edges. The project keeps the original kernel layout, channel routing, and update rule of its ASAL-derived Lenia implementation. The native experiments use a \(128\times128\) grid, radius \(R=12\), and \(T=2\).

Artificial-life research asks how simple computational rules can produce organized, persistent, or adaptive-looking behavior. That question motivates our exploration, but a simulated pattern is not thereby a living organism. Lenia qualifies as a zero-player game because, once its parameters and initial grid are fixed, every later state follows from the transition rule without player input.

## 2. How Do We Design and Search?

The project automates the choice of a Lenia configuration rather than drawing individual outcomes by hand. Its starting point is the three-channel, 15-kernel `5N7KKM` LeniaBreeder family [3], as incorporated in the upstream ASAL code [4]. A configuration is represented by 3,117 values: 45 offsets for the kernels' \(m\), \(s\), and \(h\) parameters, and 3,072 values for a \(32\times32\times3\) initial patch. The offsets are transformed into valid parameter and cell ranges before simulation. Kernel shapes, channel connections, grid size, and time scale remain fixed. Thus, the search explores a substantial part of **one** Lenia family; it does not invent arbitrary new cellular-automaton rules.

The first design route is **target-guided search**, adapted from ASAL [4]. The user supplies a text prompt. We encode the prompt and sampled simulation images with `apple/MobileCLIP2-S2` [5], then average their normalized cosine similarities across eight frames. Sep-CMA-ES, an evolutionary optimizer, proposes new configuration vectors to improve that score. The saved target experiment used the phrase **“a biological cell under the microscope”**, a population of 16, 1,000 generations, and 256 simulation steps per evaluation. Its best recorded similarity was **0.3447**. This number measures agreement in the chosen embedding space; it is not a probability that the pattern is a biological cell. The [saved preview](archive/runs/vast-58980056/cell-s2/preview.png) and [trajectory video](archive/runs/vast-58980056/cell-s2/trajectory.mp4) show the selected result.

The second route is **visual diversity search**, adapted from ASAL's illumination method [4]. Beginning with a seed configuration, each iteration mutates archived configurations and evaluates the resulting trajectories. The archive keeps configurations with less redundant visual embeddings, measured using nearest-neighbor cosine similarity. The recorded run used 10,000 iterations, 32 children per iteration, and retained **5,000 unique configurations** from **320,000 proposed mutations**. Its final mean nearest-neighbor novelty, defined as one minus cosine similarity, was **0.1076**. This metric describes separation in MobileCLIP2-S2 image-feature space, not necessarily different physical behaviors.

The two-dimensional UMAP atlas below projects the saved image embeddings for inspection. Color indicates embedding novelty; marked points identify representative trajectories. UMAP helps browse the archive but is neither the simulator nor the search objective. The repository also contains a [2,000-tile static map](archive/runs/vast-59037478/diverse2000-1024-map/index.html) and individual videos.

![UMAP projection of the 5,000-configuration Lenia illumination archive](archive/runs/vast-59037478/illumination-s2/atlas.png)

*Figure 1. UMAP projection of 5,000 archived configurations. Nearby points have similar projected image embeddings; the projection alone does not establish equivalent dynamics.*

Visual similarity alone cannot determine whether a pattern survives or moves. We therefore added a numerical screen, informed by published Lenia diversity-search work and CARL's long-horizon survival filter [6, 7]. It measures mass, occupied area, temporal variation, connected-component coherence, activity, and movement with periodic boundaries taken into account. Separate checks prevent an oscillating shape from being counted as a moving one merely because its pixels change. Thresholds such as occupied area below 20%, mass coefficient of variation below 0.35, and dominant-component mass share of at least 90% were calibrated for **this implementation**; they are not claimed as universal values from those papers.

Finally, promising configurations seeded multiscale mutations, a MAP-Elites search over behavioral descriptors, and Sep-CMA-ES refinement. A random-search baseline used equal simulation-step budgets for the main neighborhood and quality-diversity comparisons. The saved campaigns performed 500,000 neighborhood evaluations and 1,000,000 quality-diversity evaluations. These are candidate evaluations, not counts of distinct persistent structures. This mathematical stage extends the visual archive with behavioral criteria while preserving the same Lenia dynamics.

## 3. How Do We Simulate?

For any proposed configuration, the simulator inserts its initial patch into the three-channel grid and repeatedly applies the Lenia update. FFT-based periodic convolution computes the kernel potentials. JAX batches Lenia trajectories on an NVIDIA GPU; PyTorch evaluates MobileCLIP2-S2 images during the visual searches. The recorded text-guided and illumination runs used an NVIDIA RTX 6000 Ada, with simulation batches of 16 and 32 respectively. Saved [text-search settings](archive/runs/vast-58980056/cell-s2/config.json), [illumination settings](archive/runs/vast-59037478/illumination-s2/config.json), configuration vectors, random seeds, scores, checkpoints, and videos allow the runs to be inspected or resumed. Numerical screening uses the native \(128\times128\) grid and update step. Gallery frames may be enlarged for viewing, but enlarging a rendered frame does not add simulation detail.

The main saved outcomes are summarized below. The screening stages deliberately use different criteria: an archive member passing an early mass check is not automatically a robust soliton.

| Experiment or stage | Recorded setup | Recorded outcome |
|---|---|---|
| Text-guided design | Cell-under-microscope prompt; 1,000 generations; population 16; 256-step trajectories | Best MobileCLIP2-S2 cosine score **0.3447** |
| Visual illumination | 10,000 iterations; 32 children each; archive capacity 5,000 | **5,000** unique retained configurations; mean embedding novelty **0.1076** |
| Numerical archive screen | All 5,000 configurations; 5,000 native-grid steps each | **957** bounded by the initial filter; **606** passed calibrated persistence and coherence checks; **86** also passed the movement rule |
| Stronger verification | 180 candidates with 10,000-step controls and damage, noise, parameter, and initialization trials | **7** passed the prespecified robustness and localization criteria: **2 moving**, **5 oscillating** |

The 5,000-step screen also labeled 3,531 trajectories as expanding and 512 as extinct; other outcomes included transient, static, and oscillating patterns. These labels summarize measured behavior under stated thresholds. In stronger verification, each of the seven retained candidates met a 75% survival fraction in each perturbation family, as well as separate activity, morphology, and localization conditions. Twenty selected ranked candidates had 20,000-step unperturbed controls and all twenty passed that extended control; this result applies to the **selected twenty**, not to every archive member. The [per-configuration screen](archive/runs/soliton-work/soliton-results/archive_screen.csv), [verified catalog](archive/runs/soliton-work/soliton-results/catalog.csv), and [detailed results report](archive/runs/soliton-work/soliton-results/REPORT.md) retain the numerical measurements behind these counts.

The examples below illustrate the two complementary views of the system. The text-guided image is an embedding-selected configuration. The moving candidate comes from the original visual archive and later passed numerical robustness checks. Its measured movement was **0.812 native-grid pixels per update** under the periodic-aware estimator; the second verified mover measured **0.370 pixels per update**. These numbers are simulation descriptors, not physical velocities. The [moving candidate's video](archive/runs/soliton-work/soliton-results/gallery/001-visual_archive-00004686.mp4) and the [ranked gallery](archive/runs/soliton-work/soliton-results/gallery/index.html) show trajectories rather than isolated snapshots.

![Preview of the configuration selected for the biological-cell prompt](archive/runs/vast-58980056/cell-s2/preview.png)

*Figure 2. Saved preview from target-guided design. The selection criterion was image–text embedding similarity.*

![Rendered frame from the verified moving archive configuration 4686](archive/runs/soliton-work/soliton-results/gallery/001-visual_archive-00004686.png)

*Figure 3. A frame from verified moving configuration 4686. The native-grid simulation was enlarged for display; movement and robustness were measured from the simulation data.*

The experiments demonstrate that automation can find many visually distinct autonomous trajectories and a smaller set that meets explicit persistence and perturbation tests. Their limits matter. The search is restricted to one Lenia parameterization and finite evaluation horizons. Similarity scores reflect a particular vision-language model; UMAP distances are only a visualization. Numerical categories depend on calibrated thresholds, and perturbation trials sample only a limited set of disturbances. A later comparison against available Lenia catalogs did not establish that the seven candidates represent undocumented forms. Neither visual resemblance nor numerical survival proves biological life.

## 4. Conclusion

This mini-project designed and simulated zero-player games by combining an established Lenia simulator with automated configuration search. Text-guided optimization produced a configuration matching a chosen description in MobileCLIP2-S2 feature space; illumination assembled a 5,000-member visual archive; and mathematical screening identified seven candidates that satisfied stronger persistence and robustness criteria. The project contribution is a reproducible workflow connecting these design and simulation stages, with saved parameters, measurements, figures, and videos. Its findings are computational outcomes within the tested Lenia family, not claims of biological or scientific novelty.

## 5. References

1. M. Gardner, “The fantastic combinations of John Conway's new solitaire game ‘Life’,” *Scientific American*, vol. 223, no. 4, pp. 120–123, 1970. [Article](https://www.scientificamerican.com/article/mathematical-games-1970-10/).
2. B. W.-C. Chan, “Lenia—Biology of Artificial Life,” *Complex Systems*, vol. 28, no. 3, pp. 251–286, 2019. [Preprint](https://arxiv.org/abs/1812.05433).
3. Faldor and Cully, “Toward Artificial Open-Ended Evolution within Lenia using Quality-Diversity,” *LeniaBreeder*, 2024. [Project and paper](https://leniabreeder.github.io/).
4. A. Kumar *et al.*, “Automating the Search for Artificial Life with Foundation Models” (ASAL), 2024. [Preprint](https://arxiv.org/abs/2412.17799).
5. Apple, “MobileCLIP2-S2,” model documentation. [Model card](https://huggingface.co/apple/MobileCLIP2-S2).
6. G. Hamon *et al.*, “Discovering Sensorimotor Agency in Cellular Automata Using Diversity Search,” 2025. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12577699/).
7. M. Cvjetko *et al.*, “CARL: The Artificial Experimentalist,” 2026. [Project and paper](https://developmentalsystems.org/carl/).
8. N. Baid *et al.*, “Guiding Evolution of Artificial Life Using Vision-Language Models” (ASAL++), 2025. [Preprint](https://arxiv.org/abs/2509.22447). This is related work; its target-proposal method is not used here.
9. T. Michel *et al.*, “Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist: Discovering Diverse Ecosystem Dynamics,” 2025. [Preprint](https://arxiv.org/abs/2505.15998). Flow-Lenia is a distinct simulation substrate and is not used here.
