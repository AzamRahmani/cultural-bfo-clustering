# Experiment V1 Results

This report summarizes the predeclared analysis (`ANALYSIS_PLAN.md`) of the
single 240-run prototype experiment (`EXPERIMENT_V1_RUN_RECORD.md`). All
numbers come from `results/prototype_30_seed_descriptive.csv` and
`results/prototype_30_seed_paired_tests.csv`; values here are rounded.

**Scope.** These results describe the current BF and CBF prototypes with
temporary prototype settings. They are not thesis results, not a complete
Cultural Algorithm, and not evidence about BF or CBF in general.

## 1. Run Outcome

| Dataset | BF successful | CBF successful | Complete pairs |
|---|---:|---:|---:|
| Iris | 30 | 30 | 30 |
| Wine | 30 | 30 | 30 |
| Glass | 27 | 25 | 25 |
| Cancer | 30 | 30 | 30 |

All 8 failures are on Glass (see the run record). Descriptive statistics use
successful runs; paired tests use complete pairs only.

## 2. Primary Metric: `best_fitness` (lower is better)

Summed squared Euclidean distance on standardized features. Values must not be
compared across datasets.

| Dataset | BF mean (95% CI) | CBF mean (95% CI) | Median CBF − BF | CBF lower / higher | Holm p | Rank-biserial r |
|---|---|---|---:|---:|---:|---:|
| Iris | 208.44 (198.05, 218.82) | 227.20 (214.08, 240.32) | +20.03 | 8 / 22 | 0.0093 | +0.58 |
| Wine | 3486.77 (3370.06, 3603.48) | 2248.07 (2198.75, 2297.38) | −1222.09 | 30 / 0 | <0.0001 | −1.00 |
| Glass | 4730.96 (4278.68, 5183.24) | 3932.71 (3586.01, 4279.40) | −677.42 | 19 / 6 | 0.0015 | −0.75 |
| Cancer | 5465.47 (5218.02, 5712.91) | 5322.82 (5092.69, 5552.96) | −233.45 | 20 / 10 | 0.1772 | −0.29 |

Under this configuration CBF reached lower fitness than BF on Wine and Glass,
higher fitness on Iris, and no difference distinguishable at the adjusted
α = 0.05 level on Cancer.

**Asymmetry.** BF returns its final-population best. CBF returns its archived
best, which excludes bacteria dispersed in the final iteration. The two
methods therefore do not select their reported solution by the same rule.

**Context.** The single seed-42 regression run favored CBF on Iris
(152.61 versus 176.89). Across 30 predefined seeds BF was lower on Iris in 22
of 30 pairs. One seed did not represent the distribution.

## 3. Secondary Metrics (exploratory)

Holm-adjusted p-values within each metric; r is positive when CBF values tend
to be larger.

### Silhouette (higher is better)

| Dataset | BF mean | CBF mean | Holm p | r |
|---|---:|---:|---:|---:|
| Iris | 0.449 | 0.441 | 0.269 | −0.32 |
| Wine | 0.092 | 0.158 | 0.0006 | +0.75 |
| Glass | 0.310 | 0.328 | 0.269 | +0.28 |
| Cancer | 0.527 | 0.540 | 0.191 | +0.39 |

### External error rate, % (lower is better)

Labels were used only after optimization, with optimal cluster-to-class
mapping.

| Dataset | BF mean | CBF mean | Holm p | r |
|---|---:|---:|---:|---:|
| Iris | 22.89 | 21.20 | 0.574 | −0.24 |
| Wine | 52.90 | 36.01 | 0.0003 | −0.85 |
| Glass | 59.83 | 60.06 | 0.911 | +0.03 |
| Cancer | 15.75 | 13.16 | 0.574 | −0.27 |

### Cost and Rejections (lower is better)

| Dataset | Runtime BF / CBF (s) | Holm p | Fitness evaluations, median CBF − BF | Holm p | Empty-cluster rejections, median CBF − BF | Holm p |
|---|---|---:|---:|---:|---:|---:|
| Iris | 0.278 / 0.403 | <0.0001 | +162.5 | <0.0001 | −22.0 | 0.0001 |
| Wine | 0.393 / 0.518 | <0.0001 | +390.5 | <0.0001 | −20.5 | 0.0037 |
| Glass | 0.423 / 0.512 | 0.0036 | +42.0 | 0.125 | +23.0 | 0.237 |
| Cancer | 0.940 / 0.887 | 0.177 | +185.5 | <0.0001 | 0.0 | 0.317 |

Runtime was measured on one Windows machine and is evidence about this
environment only. CBF used more fitness evaluations than BF on every dataset
(significant except Glass), so lower CBF fitness is not obtained at equal
evaluation cost.

## 4. Interpretation

- **No general superiority.** The direction of the BF–CBF difference depends on
  the dataset and the metric. Wine favors CBF on fitness, silhouette, and error
  rate; Iris favors BF on fitness while silhouette and error rate show no
  adjusted difference; Glass favors CBF on fitness only; Cancer shows no
  adjusted difference on any quality metric.
- **Fitness and external accuracy diverge.** On Glass, CBF's lower fitness is
  not accompanied by a lower error rate (about 60% for both). Lower
  within-cluster distance does not imply better recovery of known classes.
- **Cost.** CBF performed more fitness evaluations, so part of any fitness
  advantage may reflect additional search effort.
- **Prototype limits.** All results depend on temporary settings (population
  20, 50 iterations, step size 0.1, temporary empty-cluster policy), the
  documented CBF archive gap, and the absence of split, merge, repair, gene
  transfer, and Jcc. They are not comparable to thesis-reported values without
  verifying equivalent methods.
- **Statistics.** Thirty seeds and Wilcoxon tests describe variability for
  this configuration; significance does not imply practical relevance.

## 5. Reproduce

```powershell
.\.venv\Scripts\python.exe analyze_prototype_results.py --output-dir <new folder>
```

The analysis validates the result fingerprint first and is deterministic; on
the original Windows environment the outputs match the committed files byte
for byte. In the Linux Docker image, confidence-interval bounds can differ in
the last floating-point digit (relative difference about 1e-15); see the
Docker section of `README.md`.
