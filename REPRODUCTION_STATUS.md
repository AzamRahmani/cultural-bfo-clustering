# Thesis Reproduction Status Audit

## Status legend

- Verified thesis fact
- Verified software behavior
- Temporary prototype setting
- Unverified interpretation
- Missing implementation
- Requires thesis verification

## Executive summary

This repository is not a complete thesis reproduction. It is a working prototype and a reproducibility study scaffold that contains validated software behavior, temporary implementation choices, and unresolved thesis-level questions.

The current status is best summarized as follows:

- Verified software behavior exists for the dataset loader, K-Means baselines, BF prototype execution, and CBF prototype execution under fixed-seed regression conditions.
- Temporary prototype settings are used in the reusable BF and CBF implementations.
- Several thesis-level procedures remain unverified or absent.
- GSA is a missing implementation.
- BH is a missing implementation.
- No GSA or BH procedure should be invented.
- The repository should not claim complete thesis reproduction.

## Baseline summary

K-Means baselines currently exist for Iris, Wine, Glass, and Cancer.

These baselines are verified software behavior in the current codebase. They are not equivalent to a full thesis reconstruction unless the original thesis procedure is explicitly matched.

## Implementation verification matrix

| Thesis item | Current implementation | Status | Evidence or file | Scientific risk | Next verification action |
|---|---|---|---|---|---|
| dataset identity | Dataset loaders and notes identify Iris, Wine, Glass, and Cancer; Cancer uses the OpenML breast-w dataset with missing-row filtering | Verified thesis fact | PROJECT_NOTES.md; data_loader.py | Low if the dataset identity is recorded correctly; moderate if the thesis uses a different preprocessing specification | Confirm exact thesis dataset names and filtering rules against the original source |
| dataset preprocessing | Missing-value handling is documented only in project notes and implemented in a filter step; exact thesis preprocessing is not proven | Requires thesis verification | PROJECT_NOTES.md; data_loader.py | High | Read the original thesis method, reproduce the exact preprocessing, and document all filtering steps |
| feature standardization | StandardScaler is applied before clustering in the reusable pipelines | Verified software behavior | data_loader.py; README.md | Low for implementation consistency; moderate if the thesis required a different scaling policy | Check whether the thesis uses standardized features, normalized features, or a different preprocessing pipeline |
| number of clusters | 3 for Iris and Wine; 6 for Glass; 2 for Cancer are implemented directly in scripts and reusable functions | Verified software behavior | compare_iris.py; compare_wine.py; evaluate_glass_seeds.py; evaluate_cancer_seeds.py; README.md | Low for code consistency; moderate if the thesis defines different cluster counts | Match the thesis cluster settings and document them explicitly |
| exclusion of known labels during optimization | Clustering code does not use y labels during optimization; labels are used only as external information or validation | Verified software behavior | data_loader.py; clustering scripts | Low | Confirm that the thesis indeed excludes labels during optimization and does not use them for post hoc mapping |
| within-cluster distance objective | The objective is based on squared Euclidean distance to cluster centers within the current assignment | Verified software behavior | bf_reusable.py; cbf_reusable.py; test_project.py | Low for implementation fidelity to the project code; moderate if the thesis uses a different objective definition | Validate the exact equation used in the thesis and compare it with the function implementation |
| BF population initialization | Population is initialized as bacteria x clusters x features with a repeatable random-seed protocol | Verified software behavior | bf_reusable.py; bf_iris.py | Low | Match the exact thesis population-creation rule and initialization distribution |
| tumble | Random directional perturbation is implemented in the BF movement logic | Verified software behavior | bf_reusable.py; bf_iris.py | Low | Confirm whether the thesis uses the same tumble operator and step distribution |
| swim | Local search through repeated perturbation and acceptance is implemented | Verified software behavior | bf_reusable.py; bf_iris.py | Low | Confirm the exact swim-length, step-size, and acceptance logic used in the thesis |
| greedy acceptance | The implementation accepts better local candidates as the search proceeds | Verified software behavior | bf_reusable.py; cbf_reusable.py | Low | Confirm whether the thesis uses the same greedy acceptance rule and not a different survival criterion |
| bacterial health | Health accumulation is tracked during the BF lifecycle and used in reproduction decisions | Verified software behavior | bf_reusable.py; bf_iris.py | Low | Confirm the thesis health definition and whether it matches this implementation |
| reproduction | Population reproduction is implemented after health accumulation and movement | Verified software behavior | bf_reusable.py; bf_iris.py | Low | Confirm the exact thesis reproduction schedule and selection rules |
| elimination-dispersal | Random dispersal and elimination are implemented in the BF lifecycle | Verified software behavior | bf_reusable.py; bf_iris.py | Low | Confirm the thesis elimination-dispersal counts, timing, and probability model |
| empty-cluster behavior | Reformulated candidate rejection is used when a solution produces fewer nonempty clusters than requested | Temporary prototype setting | README.md; bf_reusable.py; cbf_reusable.py | High | Verify the thesis policy from the original method and replace or justify this prototype policy |
| situational knowledge | Situational cultural knowledge is implemented in the CBF prototype | Verified software behavior | cbf_reusable.py; cbf_iris.py | Moderate | Confirm whether this matches the thesis cultural mechanism or is a simplification |
| normative knowledge | Normative interval knowledge is implemented for cluster bounds | Verified software behavior | cbf_reusable.py; cbf_iris.py | Moderate | Confirm the exact normative update rule and whether the thesis used the same procedural form |
| cultural influence | A cultural influence term is used in the adaptive prototype | Temporary prototype setting | README.md; cbf_reusable.py | High | Verify the thesis-defined cultural influence procedure and parameterization |
| archived best solution | The implementation tracks an archived best bacterium and best fitness across the run | Verified software behavior | cbf_reusable.py; cbf_iris.py | Low | Confirm the thesis archiving rule and whether it is a full cultural belief update or an approximation |
| stopping conditions | Fixed iteration counts and prototype lifecycle boundaries are used in the code | Temporary prototype setting | README.md; bf_reusable.py; cbf_reusable.py | High | Replace with thesis-defined stopping criteria and convergence rules |
| random-seed protocol | Reusable functions accept a random_seed parameter and fixed-seed regression runs are used | Verified software behavior | README.md; test_project.py; evaluate_wine_seeds.py | Low | Confirm the thesis seed protocol if the original method prescribes a specific randomization strategy |
| within-cluster distance reporting | Distances are reported as sum of squared distances and used as a fitness metric | Verified software behavior | README.md; test_project.py; compare_wine.py | Moderate | Confirm the exact thesis metric reporting convention and whether it uses a different normalization or objective |
| standard-deviation reporting | Standard deviations are reported for seed-based results in the evaluation scripts | Verified software behavior | evaluate_wine_seeds.py; evaluate_glass_seeds.py; evaluate_cancer_seeds.py | Low | Verify the thesis definition of sample standard deviation versus population standard deviation |
| error-rate calculation | No explicit thesis-level error-rate calculation is implemented in the documented prototype | Missing implementation | README.md; repository files | High | Reconstruct the exact error-rate definition and implement it only if the thesis defines it |
| cluster-to-class mapping | The repository does not implement a thesis-required mapping from cluster labels to classes | Missing implementation | README.md; clustering scripts | High | Only add this if the thesis specifies a cluster-to-class mapping procedure and validation rule |
| GSA comparison | No GSA procedure or comparison implementation exists in the repository | Missing implementation | repository files | High | Do not invent GSA behavior. Verify against the original thesis requirements before implementing |
| BH comparison | No BH procedure or comparison implementation exists in the repository | Missing implementation | repository files | High | Do not invent BH behavior. Verify against the original thesis requirements before implementing |
| repeated-run count | Seed sweeps exist for 5 seeds in several evaluation scripts, but this is not necessarily the thesis protocol | Requires thesis verification | evaluate_wine_seeds.py; evaluate_glass_seeds.py; evaluate_cancer_seeds.py | High | Establish the exact repeated-run count required by the thesis and document the protocol before claiming statistical validity |
| statistical testing | No final statistical test plan is implemented or documented as a thesis procedure | Requires thesis verification | README.md; evaluation scripts | High | Define the exact test procedure after the thesis reporting protocol is verified |

## Empty-cluster rejection policy

Temporary prototype setting

Candidates producing fewer nonempty clusters than requested receive infinite fitness and are rejected.

This is a temporary prototype policy. It is not established as the thesis behavior and should not be treated as a thesis fact.

## Seed-42 reusable regression references

These are verified software regression references, not universal scientific targets.

| Dataset | Method | Fitness | Silhouette | Empty-cluster rejections |
|---|---|---:|---:|---:|
| Iris | BF | 176.8929 | 0.4483 | 24 |
| Iris | CBF | 152.6087 | 0.4566 | 7 |
| Wine | BF | 3366.1483 | 0.1528 | 36 |
| Wine | CBF | 2316.0097 | 0.1722 | 6 |
| Glass | BF | 5593.8385 | 0.2135 | 534 |
| Glass | CBF | 4335.7315 | 0.2922 | 615 |
| Cancer | BF | 4763.2108 | 0.5471 | 0 |
| Cancer | CBF | 4492.6286 | 0.5649 | 0 |

These numbers are regression checks for the current implementation. They document software behavior under fixed-seed conditions and should not be interpreted as universal scientific targets.

## Prototype evaluation section

The repository includes five-seed evaluations for Iris, Wine, Glass, and Cancer.

These evaluations are useful for implementation sanity checks, but they do not establish statistical significance.

Important constraints:

- Five seeds do not establish statistical significance.
- Fitness values from different datasets are not directly comparable.
- Fitness and silhouette may favor different methods.
- No general BF or CBF superiority is established.

This means the current prototype results are descriptive only and should be treated as software-level evidence, not as a final scientific conclusion.

## Prioritized verification backlog

### Priority 1

- objective-function equation
- BF lifecycle and parameters
- Cultural Algorithm procedures
- empty-cluster behavior
- metric definitions

### Priority 2

- standard-deviation definition
- error-rate calculation
- cluster-to-class mapping
- repeated-run methodology
- GSA details
- BH details

### Priority 3

- predefined 30-seed protocol
- runtime measurement
- convergence tracking
- fitness-evaluation counts
- machine-readable result files
- statistical analysis

## Project A and Project B Boundary

- Project A is the thesis reproduction and reproducibility study.
- Project B is the modern AI extension.
- Project A should be frozen and tagged before Project B begins.
- Project B must not silently modify Project A results.
- Project B will later investigate sensor data, learned representations, anomaly detection, and edge feasibility.

## Explicit implementation limitations

- GSA is a missing implementation.
- BH is a missing implementation.
- No GSA or BH procedure should be invented.
- The repository does not claim complete thesis reproduction.
- The repository does not present temporary settings as verified thesis parameters.
- The repository does not present prototype outcomes as universal scientific claims.

## Conclusion

The repository contains a substantial prototype implementation and a useful software-level validation scaffold. It is not yet approved as a full thesis reproduction because the original thesis procedures, exact parameters, and statistical methodology remain unverified.

The correct interpretation is:

- Verified software behavior exists for the current implementation.
- Temporary prototype settings exist where the thesis procedure is not yet matched.
- Missing implementation pieces remain for GSA and BH.
- Thesis-level verification is still required before any claim of final reproduction is justified.
