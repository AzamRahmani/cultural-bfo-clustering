# Analysis Plan for Experiment V1

This plan completes the methods that `EXPERIMENT_PROTOCOL.md` Sections 10 and
11 left to be selected before analysis. It was written and committed after the
run record (`EXPERIMENT_V1_RUN_RECORD.md`) and before any metric was
summarized or compared. Only run status, failure descriptions, and the two
archive-gap values in the run record had been inspected.

## 1. Input

- `results/prototype_30_seed_results.csv`, SHA-256
  `68d407c7bb9e4824383b1122c161c0925eb11a38f867460f33abcfa88d622eb8`
- The analysis must refuse to run if the file fingerprint differs from the
  sidecar or if the Section 15 record counts are not met.

## 2. Metrics and Direction

| Metric | Role | Better direction |
|---|---|---|
| `best_fitness` | primary | lower |
| `silhouette` | secondary | higher |
| `error_rate_percent` | secondary | lower |
| `runtime_seconds` | secondary, environment-dependent | lower |
| `fitness_evaluation_count` | secondary | lower |
| `empty_cluster_rejections` | secondary | lower |

`accuracy_percent` is `100 - error_rate_percent` and is not analyzed
separately. Metrics are reported separately and are never combined into one
score.

## 3. Descriptive Statistics

Computed for each dataset, algorithm, and metric over successful runs only.
The number of failed runs is reported next to every summary.

- count, mean, median, minimum, maximum
- population standard deviation, `np.std(values, ddof=0)` (Protocol Section 10)
- first and third quartiles, `np.percentile(values, [25, 75], method="linear")`
- interquartile range, third quartile minus first quartile
- **95% confidence interval for the mean:** two-sided Student t interval,
  `mean ± t(0.975, n - 1) * s / sqrt(n)`, where `s = np.std(values, ddof=1)`
  and `t` is `scipy.stats.t.ppf`. The interval uses the sample standard
  deviation because it estimates uncertainty of the mean; the reported
  dispersion remains the population standard deviation required by the
  protocol. If `s == 0` the interval is `[mean, mean]`; if `n < 2` it is left
  empty.

The t interval assumes approximately normal sampling of the mean. It is
reported as a descriptive uncertainty interval, not as a test.

## 4. Paired BF versus CBF Comparison

- **Pairing:** within each dataset, by seed.
- **Incomplete pairs:** complete-case analysis. A seed is included only if
  both BF and CBF succeeded. Excluded seeds are listed in the output. For V1
  this excludes Glass seeds 9, 11, 13, 19, and 28 (25 Glass pairs remain);
  all other datasets have 30 pairs.
- **Difference:** `d = CBF - BF` for each pair.
- **Test:** two-sided Wilcoxon signed-rank test,
  `scipy.stats.wilcoxon(cbf, bf, zero_method="wilcox",
  alternative="two-sided", method="auto")`. It is chosen because the pairing
  is known, the sample is small, and no normality of differences is assumed.
  It was not chosen by inspecting results.
- **Effect size:** matched-pairs rank-biserial correlation over nonzero
  differences, `r = (R_plus - R_minus) / (R_plus + R_minus)`, using average
  ranks of `|d|`. Positive `r` means CBF values tend to be larger than BF
  values; whether that is better depends on the metric direction.
- **Also reported:** median paired difference, and the numbers of pairs where
  CBF is lower, higher, or tied.
- **All differences zero:** no test is performed; the p-value is left empty
  and `r = 0`.

## 5. Multiple Comparisons

Holm-Bonferroni adjustment across the four datasets, separately for each
metric, at family-wise α = 0.05. The primary metric is the confirmatory
comparison; secondary metrics are exploratory. Both raw and adjusted p-values
are reported.

## 6. Interpretation Constraints

- Fitness values are never compared across datasets.
- BF returns its final-population best; CBF returns its archived best, which
  misses bacteria dispersed in the final iteration. Any `best_fitness`
  comparison must state this asymmetry.
- Runtime depends on the machine and is evidence about this environment only.
- Failures are reported as counts per dataset and algorithm; no test is
  applied to failure rates.
- A statistically significant difference does not imply practical relevance,
  and results apply only to this prototype configuration.
- Agreement or disagreement with thesis-reported numbers is not part of this
  analysis.

## 7. No Further Analyses

No sensitivity analyses, alternative tests, subgroup analyses, or outlier
removal are planned. Any additional analysis must be labeled as post hoc.

## 8. Outputs

`analyze_prototype_results.py` writes:

- `results/prototype_30_seed_descriptive.csv`
- `results/prototype_30_seed_paired_tests.csv`
