# Experiment Protocol

## 1. Purpose and Scientific Boundary

This protocol evaluates the current reusable BF and CBF prototype behavior
across repeated random seeds. It does not establish exact thesis reproduction.
BF and CBF parameters remain temporary prototype settings, not verified thesis
parameters. The experiment design, metrics, seed list, exclusions, and analysis
rules must be frozen before results are inspected and must not be redesigned
after seeing results. Project B is outside this protocol.

## 2. Research Questions

- **RQ1:** How variable are BF and CBF fitness results across random seeds?
- **RQ2:** How variable are silhouette and externally evaluated error rates?
- **RQ3:** How frequently do empty-cluster candidates occur?
- **RQ4:** Do fitness, silhouette, and error rate favor the same method?
- **RQ5:** What runtime and fitness-evaluation cost does each method require?

## 3. Datasets and Cluster Counts

| Dataset | Samples | Features | Requested clusters |
|---|---:|---:|---:|
| Iris | 150 | 4 | 3 |
| Wine | 178 | 13 | 3 |
| Glass | 214 | 9 | 6 |
| Cancer | 683 | 9 | 2 |

Known class labels are excluded from optimization. They are used only after an
optimization run has completed, for external error-rate evaluation using the
cluster-to-class assignment in `clustering_error_rate`.

## 4. Preprocessing

The current loader applies `StandardScaler` to each dataset's feature matrix
and supplies the resulting `X_scaled` to the reusable algorithms. The Cancer
loader first removes rows with missing feature values and then standardizes
the remaining data. This describes current software behavior; the exact thesis
preprocessing remains **Requires thesis verification**.

Within each dataset experiment, BF and CBF must receive the same standardized
feature matrix. Known labels must not be included in preprocessing used by the
optimization algorithms.

## 5. Algorithms

- Reusable BF: `run_bf` in `bf_reusable.py`.
- Reusable CBF: `run_cbf` in `cbf_reusable.py`.
- Historical standalone Iris prototypes are excluded.

K-Means may be reported as a separate deterministic baseline using its current
fixed `random_state` configuration. It is not a stochastic repeated-seed
algorithm under that configuration and is not included in the 240 stochastic
BF/CBF result rows.

## 6. Fixed Configuration

The following values are the current reusable function defaults, taken from
the `run_bf` and `run_cbf` signatures. All are **Temporary prototype
settings**; no thesis parameter is inferred from them. The experiment uses the
dataset-specific cluster counts in Section 3 and explicitly overrides the
random seed according to Section 7.

| Parameter | Reusable BF default | Reusable CBF default |
|---|---:|---:|
| `random_seed` | 42 | 42 |
| `population_size` | 20 | 20 |
| `num_iterations` | 50 | 50 |
| `step_size` | 0.1 | 0.1 |
| `max_swim_steps` | 3 | 3 |
| `reproduction_interval` | 10 | 10 |
| `elimination_interval` | 10 | 10 |
| `elimination_probability` | 0.10 | 0.10 |
| `cultural_influence` | Not applicable | 0.05 |
| `acceptance_ratio` | Not applicable | 0.35 |

`X_scaled` and `num_clusters` are required inputs rather than defaults. Current
reusable BF/CBF reject candidates that fail to produce all requested
nonempty clusters; this is a temporary prototype policy, not verified thesis
behavior. No parameters may be tuned during this experiment.

## 7. Seed Protocol

The predeclared seed list is exactly:

```text
seeds = 0 through 29
```

All 30 seeds must be executed for every dataset and both algorithms. Failed or
unfavorable runs must not be removed or replaced with another seed. BF and CBF
must use the same seed list for paired comparisons. Results must not be
replaced through post-selection. Seed 42 remains a software regression
reference and is not part of seeds 0 through 29 unless separately reported.

## 8. Required Per-Run Fields

Each dataset-algorithm-seed run must record the following fields. “Available”
means the current reusable result already returns the value or it is computed
from returned values by the evaluation runner. Any field not currently
returned by the reusable implementation is explicitly marked **Requires
implementation**.

| Field | Current availability |
|---|---|
| dataset | Available from experiment configuration |
| algorithm | Available from experiment configuration |
| seed | Available from run configuration/result |
| `best_fitness` | Available |
| `final_population_best_fitness` | Available for CBF; not returned for BF |
| silhouette | Available |
| error-rate percentage | Computed after optimization with the external evaluator |
| accuracy percentage | Computed after optimization with the external evaluator |
| correct and incorrect counts | Computed after optimization with the external evaluator |
| nonempty cluster count | Computed from returned labels after optimization |
| `empty_cluster_rejections` | Available |
| `runtime_seconds` | **Requires implementation**; measure wall-clock time around the optimizer call, excluding dataset loading and external evaluation |
| `fitness_evaluation_count` | **Requires implementation**; not present in either returned result |
| accepted movement count | Available as `accepted_movements` |
| reproduction event count | Available as `reproduction_events` |
| elimination event count | Available as `elimination_events` |
| dispersed bacterium count | Available as `dispersed_bacteria_count` |
| parameter configuration | **Requires implementation** in the result record; record all explicit and defaulted arguments used |
| Python/package version information | **Requires implementation** in run metadata; record Python, NumPy, scikit-learn, SciPy, and Matplotlib versions |

The term **Requires implementation** identifies a result or metadata field
that the current reusable algorithms do not return. It does not authorize
algorithm changes in this documentation milestone.

## 9. Machine-Readable Output

The future CSV path is:

```text
results/prototype_30_seed_results.csv
```

The CSV must contain one row per dataset × algorithm × seed combination. The
expected successful design is 4 datasets × 2 algorithms × 30 seeds = 240
stochastic result rows. Failed attempts must also be represented rather than
silently omitted; include an explicit run status and failure description, with
unavailable metrics left empty rather than substituted. Do not create the
directory or CSV as part of this protocol milestone.

## 10. Aggregate Statistics

For each dataset, algorithm, and metric, predefine reporting:

- count
- mean
- population standard deviation, using `np.std(..., ddof=0)`
- median
- minimum
- maximum
- first quartile (25th percentile)
- third quartile (75th percentile)
- interquartile range (third quartile minus first quartile)
- 95% confidence interval

Use the NumPy linear percentile method explicitly for quartiles so the
calculation is reproducible. The exact 95% confidence-interval method remains
to be selected before implementation and must be documented explicitly before
results are analyzed.

## 11. Comparisons

BF and CBF comparisons must be paired within each dataset because both methods
use the same seed list. The exact statistical test and effect-size measure
remain to be selected before results are analyzed and must be documented
before analysis. Do not select a test merely because it produces statistical
significance. Any method selection must account for the pairing and the
predeclared research questions.

## 12. Primary and Secondary Metrics

The primary prototype metric is `best_fitness`.

Secondary metrics are:

- silhouette
- externally evaluated error rate
- runtime
- fitness-evaluation count
- empty-cluster rejections

Fitness values from different datasets must not be compared directly because
the datasets have different scales, dimensions, and sample counts. Fitness,
silhouette, and error rate may favor different methods; report each metric
without treating one as a substitute for another.

## 13. Failure Policy

- Do not silently remove failed runs.
- Do not use a replacement seed.
- Record each failure, its seed, dataset, algorithm, and failure details.
- Report non-finite results explicitly; do not silently discard or repair them.
- Report incorrect cluster counts explicitly; do not silently relabel them as
  successful runs.
- Do not repair or tune an algorithm after seeing experimental results under
  this protocol. Any such change requires a new protocol version and a new
  experiment.

## 14. Reproducibility Metadata

Record with the result set, and associate with each run in the CSV or
unambiguously linked metadata:

- Git commit hash
- operating system
- Python version
- NumPy version
- scikit-learn version
- SciPy version
- Matplotlib version
- exact command used
- run start and completion timestamps
- parameter configuration

Environment and command metadata should identify the exact software state that
produced the runs. No package version is inferred from this protocol.

## 15. Output Validation

Before analysis, validate that:

- There are exactly 240 stochastic run records: 4 datasets × 2 algorithms ×
  30 seeds, with failed attempts retained as records.
- Dataset-algorithm-seed combinations are unique.
- Every dataset-algorithm pair has all seeds 0 through 29; no seed is missing.
- Metrics are finite for runs reported as successful; non-finite metrics are
  explicitly flagged and reported as failures or invalid outputs.
- Nonempty cluster counts and requested cluster counts are reported and
  checked for each run.
- Known labels were excluded from optimization and used only for external
  error-rate evaluation after each run.
- For CBF, archived-best consistency is checked: returned `centers` exactly
  equal `belief_best_bacterium`, and `best_fitness` is no greater than
  `final_population_best_fitness`.

## 16. Interpretation Rules

- Thirty seeds improve the evidence about prototype variability but do not
  automatically establish universal superiority.
- Statistical significance does not automatically imply practical relevance.
- Negative and inconclusive results must be reported.
- Software tests do not prove scientific validity.
- Prototype findings must not be described as thesis reproduction results.

## 17. Versioning Boundary

This is protocol version 1 for the preserved reusable prototype. Later
thesis-compatible algorithm changes require a new protocol version. Project B
requires a separate experiment protocol.

## 18. Implementation Backlog

Future focused milestones:

- add runtime measurement
- add fitness-evaluation counters
- create the CSV result writer
- implement the 30-seed runner
- validate the CSV
- select and document statistical methods before analysis
- generate plots only after raw results are frozen