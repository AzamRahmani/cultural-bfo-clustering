# Project B Experiment Protocol (Draft)

**Status: DRAFT, not frozen.** Section 15 lists the decisions that are still
open. Once they are settled, this protocol is frozen with a Git tag, before any
model is trained and before any statistic of the data is computed. After
freezing, the design must not be changed in response to results; a change
requires a new protocol version and a new experiment.

## 1. Purpose and Boundary

Project B studies whether learned representations of raw smartphone sensor
signals help (a) unsupervised clustering of activities and (b) detection of
unseen activities, and whether the learned models are small and fast enough
for edge devices. It reuses the frozen Project A clustering code unchanged and
never modifies Project A files or results (`project-a-v1`).

## 2. Research Questions

- **RQ1 (representation for clustering):** Does clustering an autoencoder
  representation of raw inertial signals recover activity classes better than
  clustering (a) the provider's 561 handcrafted features, (b) the raw signals,
  and (c) a PCA representation of the same dimension?
- **RQ2 (anomaly detection):** When one activity is withheld during training,
  does autoencoder reconstruction error detect windows of that unseen activity
  better than PCA reconstruction error and an Isolation Forest?
- **RQ3 (edge feasibility):** What are the parameter count, model size, and
  single-thread inference latency of the learned encoder, compared with PCA?
- **RQ4 (algorithm transfer, exploratory):** Do the Project A BF and CBF
  prototypes behave on HAR representations as they did on the small benchmark
  datasets, relative to K-Means?

## 3. Data

UCI HAR, recorded in `PROJECT_B_DATA.md` (SHA-256 fingerprints). The loader
must verify the outer zip fingerprint before use and refuse to run otherwise.

- **Split:** the provider's subject-wise split: 7,352 training windows
  (21 subjects) and 2,947 test windows (9 different subjects). The test
  subjects never contribute to fitting any scaler, PCA, autoencoder, Isolation
  Forest, or clusterer.
- **Classes:** 6 activities. Labels are never used to fit representations or
  clusterers. In RQ2, training labels are used only to remove the withheld
  activity from the training data; this is part of the novelty-detection
  design.
- **Raw signal input:** the 9 inertial channels (body acceleration, body
  gyroscope, total acceleration; x, y, z), 128 samples per window, shape
  9 × 128.

## 4. Preprocessing

All statistics come from the fitting set (training windows only; in RQ2,
training windows without the withheld activity).

- **HAND:** the 561 provider features, standardized per feature
  (`StandardScaler`, fitted on the fitting set).
- **RAW:** each of the 9 channels standardized with its own mean and standard
  deviation over all fitting windows and time steps; flattened to 1,152 values
  for clustering.
- **PCA-L:** PCA with L components fitted on standardized, flattened RAW
  (`svd_solver="full"`, deterministic).
- **AE-L:** encoder output (L values) of the autoencoder in Section 5, applied
  to standardized RAW.

L is the latent dimension, fixed at **16** (open decision D1).

## 5. Autoencoder (fixed; no tuning)

1-D convolutional autoencoder in PyTorch, CPU only.

| Part | Layers |
|---|---|
| Encoder | Conv1d(9→32, kernel 5, stride 2, padding 2), ReLU; Conv1d(32→64, kernel 5, stride 2, padding 2), ReLU; Flatten (64 × 32); Linear(2048→L) |
| Decoder | Linear(L→2048), ReLU; reshape 64 × 32; ConvTranspose1d(64→32, kernel 4, stride 2, padding 1), ReLU; ConvTranspose1d(32→9, kernel 4, stride 2, padding 1) |

- Loss: mean squared error on standardized RAW.
- Optimizer: Adam, learning rate 1e-3, batch size 128, **30 epochs**
  (open decision D2), shuffling with the run seed.
- No early stopping, no validation-based selection, no hyperparameter search.
  The model after the final epoch is used.
- Determinism: `torch.manual_seed(seed)`, NumPy seeded,
  `torch.use_deterministic_algorithms(True)`, a fixed thread count recorded in
  the metadata.

## 6. Clustering (RQ1, RQ4)

Clustering is performed on the 2,947 **test** windows, represented by each
representation fitted on the training windows. k = 6.

| Algorithm | Configuration |
|---|---|
| K-Means | scikit-learn, `n_init=10`, `random_state=seed` |
| BF | `run_bf` from `bf_reusable.py`, Project A defaults (`EXPERIMENT_PROTOCOL.md` Section 6), `random_seed=seed` |
| CBF | `run_cbf` from `cbf_reusable.py`, Project A defaults, `random_seed=seed` |

Unlike Project A, K-Means is run with every seed, so it is a paired
stochastic baseline. BF and CBF are applied to all four representations
(open decision D3: they may be slow and weak in 561 and 1,152 dimensions).

## 7. Anomaly Detection (RQ2)

Leave-one-activity-out: for each activity a ∈ {1, …, 6} and each seed:

1. Fitting set = training windows whose activity is not a.
2. Fit preprocessing, the autoencoder, PCA-L, and an Isolation Forest on the
   fitting set.
3. Score every test window (all 6 activities). Positive class = activity a.

| Detector | Anomaly score |
|---|---|
| AE | mean squared reconstruction error over 9 × 128 values |
| PCA | mean squared reconstruction error of PCA-L |
| IF | Isolation Forest on HAND features, `n_estimators=100`, `random_state=seed`; score = `-score_samples` |

PCA is deterministic and therefore gives the same value for every seed.

## 8. Edge Measurements (RQ3)

For the RQ1 encoders (seed 0 only, since the architecture fixes size and the
cost does not depend on the seed) and for PCA-L:

- parameter count and float32 weight size in bytes;
- ONNX file size after export;
- latency per single window with ONNX Runtime, `intra_op_num_threads=1`,
  100 warm-up runs, then the median and 95th percentile over 1,000 runs.

Latency on this PC is a proxy, not a phone measurement, and is reported as
environment-dependent. The cost of computing the 561 handcrafted features
cannot be measured: the provider's feature-extraction code is not in the
dataset. Quantization is out of scope (open decision D4).

## 9. Seeds and Run Count

Seeds 0 through 29, all executed, none replaced. The seed controls the
autoencoder initialization and shuffling, K-Means, BF, CBF, and the Isolation
Forest within a run.

| Part | Records |
|---|---:|
| RQ1/RQ4 clustering | 4 representations × 3 algorithms × 30 seeds = 360 |
| RQ2 anomaly | 6 withheld activities × 3 detectors × 30 seeds = 540 |
| RQ3 edge | 2 models (AE-16 encoder, PCA-16), measured once each |

Autoencoder trainings: 30 (RQ1) + 180 (RQ2) = 210.

## 10. Metrics

| Metric | RQ | Role | Better |
|---|---|---|---|
| clustering error rate, optimal one-to-one mapping (`clustering_error_rate`) | RQ1 | **primary** | lower |
| adjusted Rand index, normalized mutual information | RQ1 | secondary | higher |
| silhouette in the representation's own space | RQ1 | secondary; not comparable across representations | higher |
| BF/CBF fitness, runtime, fitness-evaluation count | RQ4 | exploratory | lower |
| ROC-AUC | RQ2 | **primary** | higher |
| PR-AUC (average precision) | RQ2 | secondary | higher |
| autoencoder final training loss, training time | all | descriptive | – |
| parameters, bytes, latency | RQ3 | descriptive | lower |

## 11. Statistics

The methods of `ANALYSIS_PLAN.md` are reused: descriptive statistics with a
Student t 95% confidence interval; paired two-sided Wilcoxon signed-rank tests
by seed; matched-pairs rank-biserial effect size; Holm-Bonferroni correction.

**Confirmatory comparisons** (one Holm family, α = 0.05):

- RQ1, K-Means, error rate: AE-16 vs PCA-16, AE-16 vs HAND, AE-16 vs RAW.
- RQ2, ROC-AUC averaged over the 6 withheld activities per seed: AE vs PCA,
  AE vs IF.

Everything else, including per-activity RQ2 results and all BF/CBF results,
is exploratory and labeled as such.

## 12. Failure Policy

As in `EXPERIMENT_PROTOCOL.md` Section 13: failed runs are recorded with
their details, never removed or replaced; non-finite losses or scores are
reported as failures; nothing is tuned or repaired after results are seen.

## 13. Outputs and Metadata

Results go under `results/project_b/`, never into Project A files:
`clustering_results.csv`, `anomaly_results.csv`, `edge_results.csv`, each
with a SHA-256 sidecar. Metadata: Git commit, protocol tag, operating system,
CPU model, thread counts, Python, NumPy, scikit-learn, SciPy, PyTorch, ONNX
Runtime versions, the exact command, timestamps, and the full configuration.

## 14. Interpretation Rules

- Results hold for HAR, this architecture and these settings only.
- A better clustering error rate does not mean the representation is better
  for every task.
- Novelty detection of a withheld activity is a proxy for anomaly detection,
  not evidence about real sensor faults.
- Latency on a PC is not a phone measurement.
- Negative and inconclusive results are reported.

## 15. Open Decisions Before Freezing

| ID | Decision | Draft choice |
|---|---|---|
| D1 | Latent dimension L | 16 |
| D2 | Training epochs | 30, fixed |
| D3 | Run BF/CBF on all representations, or only on AE-16 and PCA-16 | all four |
| D4 | Include int8 quantization in RQ3 | no |
| D5 | Add PyTorch and ONNX Runtime to `requirements.txt` (latest available: torch 2.15.0, onnxruntime 1.31.1) and to the Docker image | yes |

## 16. Implementation Backlog

After freezing, each a focused milestone with tests:

1. HAR loader with fingerprint check and leakage-safe preprocessing
2. autoencoder module and deterministic training
3. clustering runner (RQ1/RQ4) and CSV writer
4. anomaly runner (RQ2)
5. edge measurements (RQ3)
6. output validation, then analysis per Section 11
