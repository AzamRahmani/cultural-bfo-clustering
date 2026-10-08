# Structural Operations Audit

## 1. Audit Purpose

This audit examines thesis evidence and current software behavior for:

- replacement
- split or division
- merge
- repair
- gene transfer
- dynamic cluster count
- maximum cluster count

This is a documentation-only audit. No implementation decision is authorized.

## 2. Thesis Evidence Inventory

The thesis evidence already recorded in the project audits states:

- Replacement probability: `0.60` (parameter evidence recorded from page 67).
- Split probability: `0.10` (parameter evidence recorded from page 67).
- Merge probability: `0.10` (parameter evidence recorded from page 67).
- Maximum cluster count: `10` (parameter evidence recorded from page 67).
- Pages 60-62 describe bacteria as encoding cluster assignments or other cluster-related information, and include replacement, division/splitting, merging, repair, and gene transfer. The replacement operation changes encoded cluster-related information, but its exact target and value-generation rule are unresolved.
- Splitting may create a new cluster.
- Merging combines clusters.
- Repair handles invalid candidates after structural operations.
- Gene transfer moves information from successful bacteria or belief-space knowledge to another candidate.

These are evidence statements only. The audits do not establish equations, selection rules, chromosome formats, or operation timing.

## 3. Replacement Audit

**Thesis evidence:** The recorded thesis parameters include a replacement probability of `0.60`. Replacement changes encoded cluster-related information, but the exact encoded field, target, and replacement-value generation are unresolved.

**Current implementations:** None of the four implementations has a verified equivalent of this encoded-information replacement operation.

| Implementation | Equivalent thesis replacement operation? | Current behavior relevant to replacement |
|---|---|---|
| Historical Iris BF (`bf_iris.py`) | No | Elimination-dispersal can replace a whole bacterium with a newly sampled center matrix at probability `0.10`; this is population replacement, not replacement of encoded cluster-related information. |
| Historical Iris CBF (`cbf_iris.py`) | No | Same whole-bacterium random elimination-dispersal replacement; no encoded-field replacement. |
| Reusable BF (`bf_reusable.py`) | No | Elimination-dispersal samples a new center matrix and assigns it to a population member; invalid samples may be retried. |
| Reusable CBF (`cbf_reusable.py`) | No | Same population-level random replacement; CBF belief-space influence is a movement direction, not encoded-field replacement. |

Under the current center-matrix representation, a bacterium is a `(k, d)` array of center coordinates. Any replacement expressed in this representation would have to identify whether it replaces a center row, one or more center coordinates, or some other information, and how replacement values are generated. Labels and objective/health values may depend on the changed centers and would need to be reconciled. The thesis evidence does not map its encoded cluster-related information to these center-matrix elements, so this mapping is currently unverified. This audit does not translate replacement into center perturbations.

## 4. Split Audit

The thesis evidence says division/splitting may create a new cluster and records split probability `0.10`. The precise trigger, selection rule, and initialization of that cluster are not established.

All current implementations keep the requested cluster count fixed for a run. Their movement, reproduction, and population-level elimination-dispersal preserve each bacterium's center-array shape. **Split classification: Missing implementation.**

If a split increased the current center count from `k` to `k + 1`, consequences would include:

- **Center arrays:** Each affected center matrix would need another row, changing shape from `(k, d)` to `(k + 1, d)`.
- **Population:** A regular population tensor currently shaped `(p, k, d)` could no longer contain a mixed population of old- and new-length candidates without a defined normalization step or a variable-length representation.
- **Labels:** Labels remain one per sample in shape `(n,)`, but assignments may use a new cluster identifier. Reassignment, identifier validity, and consistency with the expanded centers require defined rules.
- **Jcc dimensionality:** The standalone Jcc utility accepts flat position vectors and a population matrix, whereas current candidates are center matrices. A mapping from the changed candidate representation to Jcc positions is not verified; if a flattened-center mapping were chosen, its dimension would change with `k`, but that mapping is not authorized or established.
- **Repair:** A specified repair procedure would need to restore a valid candidate and consistently update its representation, labels, and dependent objective/health state after the split. The required sequence is not known.

## 5. Merge Audit

The thesis evidence says two clusters may be combined and records merge probability `0.10`. It does not establish a merge trigger, pair-selection rule, or representation for the combined cluster.

All current implementations preserve the requested cluster count; none reduces it during a run. **Merge classification: Missing implementation.**

If a merge reduced the center count from `k` to `k - 1`, consequences would include:

- **Center arrays:** Affected matrices would need shape `(k - 1, d)` rather than `(k, d)`.
- **Labels:** Labels remain one per sample, but assignments referring to the removed/combined cluster require a defined remapping and validity rule.
- **Population:** A mixed set of `(k, d)` and `(k - 1, d)` candidates cannot occupy the current regular `(p, k, d)` tensor without a defined normalization step or a variable-length representation.
- **Jcc:** As with splitting, current center matrices have no verified mapping to the standalone Jcc vector representation. Any representation change could alter the dimensionality used by Jcc; its mapping and behavior after a merge are unknown.
- **Repair:** A specified repair procedure would need to restore a valid candidate and consistently update its representation, labels, and dependent objective/health state after the merge. The required sequence is not known.

## 6. Repair Audit

Current empty-cluster handling is distinct from thesis structural repair:

- Reusable BF and CBF reject candidates that produce fewer nonempty clusters than the requested number by assigning infinite fitness.
- During elimination-dispersal, reusable BF and CBF may regenerate an invalid random bacterium, retrying up to 100 times to find a valid one.
- This is a **Temporary prototype setting**; it is not verified as the thesis repair procedure.
- Structural repair after split or merge is not implemented.

The following repair questions remain open; this audit does not answer them:

- How are empty clusters handled?
- How are duplicate centers handled?
- How are invalid assignment identifiers handled?
- How is chromosome length repaired?
- How are minimum and maximum cluster counts enforced?

## 7. Gene-Transfer Audit

The current Python code was searched for `gene`, `transfer`, `donor`, `recipient`, `copy`, `exchange`, and `knowledge transfer`. No algorithmically relevant donor-to-recipient gene-transfer operation was found. Relevant exact code patterns are:

- Reproduction duplicates selected bacteria: `best_bacteria = bacteria[best_indices].copy()` and `np.concatenate([best_bacteria, best_bacteria.copy()], axis=0)` in historical and reusable BF/CBF (`bf_iris.py`, `cbf_iris.py`, `bf_reusable.py`, `cbf_reusable.py`). These are copies of healthier/selected population members, not transfer of selected information between candidates.
- CBF archived-best storage copies a whole best bacterium, e.g. `belief_best_bacterium = bacteria[belief_best_index].copy()` in both CBF implementations and an update from the current best in `cbf_iris.py` / `cbf_reusable.py`. This is archived-best storage, not gene transfer.
- CBF cultural influence uses the difference between `belief_best_bacterium` and the current bacterium to form a movement direction (`cultural_direction = belief_best_bacterium - bacteria[index]` in the CBF implementations). This is cultural influence movement, not transfer of selected information into a recipient candidate.
- Elimination-dispersal replaces a population member with a newly sampled random center matrix (`bacteria[i] = rng.uniform(...)` historically; reusable code samples `replacement = rng.uniform(...)` and assigns `bacteria[index] = replacement`). This is population replacement during elimination-dispersal, not donor-to-recipient transfer.

Other textual `copy` matches (such as array conversion with `copy=False`) are not algorithmically relevant. The standalone bacterial-interaction utility and its test do not implement gene transfer. Thesis-equivalent gene transfer is **Missing implementation**.

## 8. Dynamic Cluster-Count Audit

Cluster count is fixed within each current run. Historical Iris scripts set it to three directly. Reusable BF/CBF accept a requested `num_clusters`, but preserve that value in the shapes of center arrays and candidates throughout the run; callers may select different counts across separate runs. The reusable validation that `num_clusters` not exceed the sample count is not a maximum-cluster-count-10 constraint.

| Implementation | Cluster count fixed or variable | Evidence or code location | Supports split | Supports merge | Supports maximum-cluster constraint | Status | Scientific risk |
|---|---|---|---|---|---|---|---|
| Historical Iris BF | Fixed at 3 | `bf_iris.py`: `num_clusters = 3`; population initialized with shape `(population_size, num_clusters, features)` | No (Missing implementation) | No (Missing implementation) | No; no maximum of 10 is enforced | Verified software behavior; Missing implementation | Structural operations could change representation and behavior; the fixed Iris setting is not evidence of the thesis maximum-cluster rule. |
| Historical Iris CBF | Fixed at 3 | `cbf_iris.py`: `num_clusters = 3`; population and belief/normative state retain that center shape | No (Missing implementation) | No (Missing implementation) | No; no maximum of 10 is enforced | Verified software behavior; Missing implementation | Same structural and thesis-constraint gap; cultural state also assumes the fixed center shape. |
| Reusable BF | Fixed per run at caller-requested `num_clusters` | `bf_reusable.py`: `run_bf`; initialization and replacement arrays use `(num_clusters, number_of_features)`; no structural count update | No (Missing implementation) | No (Missing implementation) | No; checks against sample count, not the thesis maximum of 10 | Verified software behavior; Missing implementation | Requested count is constrained to remain fixed; no thesis-supported dynamic-count or maximum behavior is established. |
| Reusable CBF | Fixed per run at caller-requested `num_clusters` | `cbf_reusable.py`: `run_cbf`; candidate, belief, and normative arrays retain the requested shape | No (Missing implementation) | No (Missing implementation) | No; checks against sample count, not the thesis maximum of 10 | Verified software behavior; Missing implementation | Same fixed-count gap, with belief and normative state also tied to the requested shape. |

## 9. Parameter-Use Audit

The thesis evidence records these parameters. None currently exists as the corresponding operation parameter in any of the four implementations.

| Thesis parameter | Historical BF | Historical CBF | Reusable BF | Reusable CBF |
|---|---|---|---|---|
| Replacement probability `0.60` | No | No | No | No |
| Split probability `0.10` | No | No | No | No |
| Merge probability `0.10` | No | No | No | No |
| Maximum cluster count `10` | No | No | No | No |

The historical and reusable implementations have an elimination-dispersal probability of `0.10`; that controls population dispersal and is not equivalent to replacement probability `0.60`, split probability `0.10`, or merge probability `0.10`. A configured `num_clusters` (including the historical value 3) is not the thesis maximum-cluster constraint.

## 10. Test Coverage

`test_project.py` currently protects:

- fixed center, population, belief, and normative array shapes for tested runs
- fixed label shapes
- requested nonempty cluster counts
- finite fitness values
- reusable BF/CBF empty-cluster rejection counters and fixed-seed results
- archived CBF consistency, including belief-best fitness/center relationships

The test also exercises the standalone bacterial-interaction utility, but does not integrate it into either clustering algorithm or test gene transfer.

`test_project.py` does not protect:

- replacement semantics
- split semantics
- merge semantics
- structural repair
- variable cluster count
- maximum-cluster enforcement
- gene transfer
- thesis fidelity

## 11. Required Thesis Evidence

Before implementation, the following information must be verified or explicitly documented as an unavoidable interpretation:

- candidate chromosome structure
- replacement target and replacement-value generation
- split trigger
- split selection rule
- new-cluster initialization after split
- merge trigger
- merge-pair selection
- merged-cluster representation
- repair sequence
- gene-transfer donor selection
- gene-transfer recipient selection
- transferred information
- operation ordering
- interaction with belief space
- operation probabilities and whether they are independent
- minimum number of clusters
- maximum-cluster enforcement
- objective and health recalculation after structural change

## 12. Decision Gate

Structural implementation is blocked until the representation and operation semantics are verified or explicitly documented as unavoidable interpretations. No thesis operation should be silently translated into center perturbations.

## 13. Recommended Architecture Boundary

- Do not modify `bf_reusable.py`.
- Do not modify `cbf_reusable.py`.
- Preserve current regression references.
- Future thesis-compatible structural operations require a separate module.
- Variable-length candidates may require a representation different from a regular three-dimensional NumPy population tensor.
- No architecture should be selected by this audit.
- Project B remains outside this milestone.

## 14. Safety Boundary

- No algorithm implementation is authorized.
- No missing equations or operation rules may be invented.
- No current tests or regression values may be changed.
- The private thesis PDF must not be committed.

## 15. Verification

The required final worktree check is `git status --short`. Expected output, starting from the clean worktree verified before this audit:

```text
?? STRUCTURAL_OPERATIONS_AUDIT.md
```

No Git write commands are authorized or used.