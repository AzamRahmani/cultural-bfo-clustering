# Project B Analysis Plan

This plan makes `PROJECT_B_PROTOCOL.md` Section 11 executable. It was written
and committed on 2026-11-06 while the experiment was running and before any
result value had been seen; only SUCCESS/FAILED progress lines had been
looked at. The statistical methods are those of `ANALYSIS_PLAN.md` (Project A).

## 1. Inputs

- `results/project_b/clustering_results.csv` (360 rows)
- `results/project_b/anomaly_results.csv` (540 rows)
- `results/project_b/edge_results.csv` (2 rows)

The analysis refuses to run if a file's SHA-256 differs from its metadata
sidecar, if a sidecar does not mark the full protocol design, or if the
record counts and key combinations are not exactly as in protocol Section 9.

## 2. Descriptive Statistics

Over successful runs, with the failed-run count next to every summary:
count, mean, population standard deviation (`ddof=0`), median, minimum,
maximum, linear-method quartiles, interquartile range, and the Student t 95%
confidence interval for the mean (sample standard deviation), exactly as in
`ANALYSIS_PLAN.md` Section 3.

- Clustering: per representation × algorithm for `error_rate_percent`, `ari`,
  `nmi`, `silhouette`, `fitness`, `clustering_runtime_seconds`,
  `fitness_evaluation_count`, `empty_cluster_rejections`,
  `ae_final_training_loss`, `ae_training_seconds` (the last two for AE16 only).
- Anomaly: per withheld activity × detector, and for the per-seed mean over
  the six activities (activity `ALL`), for `roc_auc`, `pr_auc`,
  `detector_runtime_seconds`.
- Edge: the two rows are reported as measured; no statistics.

## 3. Paired Comparisons

- Pairing by seed; complete-case analysis (a seed is used only if both
  compared runs succeeded; excluded seeds are listed).
- Difference `d = A - B`, where A is the first named method (always AE16 or
  AE in RQ1/RQ2).
- Two-sided Wilcoxon signed-rank test, `zero_method="wilcox"`,
  `method="auto"`; matched-pairs rank-biserial effect size; median
  difference; counts of A lower / higher / tied. If all differences are zero,
  no test is run and `r = 0`.
- For activity `ALL`, a seed's value is the mean over the six withheld
  activities and is defined only if all six runs of that detector succeeded.

## 4. Families and Multiple Comparisons

Holm-Bonferroni within each family, α = 0.05; raw and adjusted p-values are
reported.

| Family | Role | Comparisons |
|---|---|---|
| C | confirmatory | K-Means `error_rate_percent`: AE16−PCA16, AE16−HAND, AE16−RAW; activity `ALL` `roc_auc`: AE−PCA, AE−IF (5 tests) |
| E1 | exploratory | BF and CBF `error_rate_percent`: AE16 vs PCA16, HAND, RAW (6) |
| E2 | exploratory | K-Means `ari` and `nmi`: AE16 vs PCA16, HAND, RAW (6) |
| E3 | exploratory | per-activity `roc_auc`: AE vs PCA, AE vs IF (12); activity `ALL` `pr_auc`: AE vs PCA, AE vs IF (2) |
| E4 | exploratory, RQ4 | `error_rate_percent` per representation: BF−KMEANS, CBF−KMEANS, CBF−BF (12) |

Better direction: lower for error rate, fitness, runtimes, counts; higher for
ARI, NMI, silhouette, ROC-AUC, PR-AUC. Silhouette and fitness are not
compared across representations, because they are measured in different
spaces.

## 5. Interpretation Constraints

Those of protocol Section 14. In addition: PCA anomaly scores are
deterministic, so its 30 values per activity are identical and the test
compares AE's seed variability against a constant; runtimes are evidence about
this PC only.

## 6. No Further Analyses

No sensitivity or subgroup analyses and no outlier removal. Anything else is
labeled post hoc.

## 7. Outputs

`analyze_project_b.py` writes to `results/project_b/analysis/`:
`clustering_descriptive.csv`, `anomaly_descriptive.csv`, `paired_tests.csv`.
