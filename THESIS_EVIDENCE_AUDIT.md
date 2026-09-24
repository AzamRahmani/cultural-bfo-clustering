# Thesis Evidence Audit Part 1

## 1. Audit scope

This audit covers the following evidence from the original thesis PDF:

- objective function
- BF lifecycle
- candidate representation
- Cultural Algorithm components
- structural mutations
- parameters
- evaluation metrics
- benchmark results
- internal thesis inconsistencies

This document is limited to evidence extraction and classification. It does not authorize implementation changes.

## 2. Objective-function evidence

The thesis evidence states on page 66 that the intra-cluster distance is defined as the sum of ordinary Euclidean distances:

D = sum over clusters and samples of ||x_i - c_j||

This is a thesis-level fact and must be distinguished from the current project implementation.

The current project primarily uses squared Euclidean distance in its optimization objective. This is a current software behavior and not a thesis-level conclusion.

Classification:

- Verified thesis fact: The thesis defines intra-cluster distance as ordinary Euclidean distance on page 66.
- Verified software behavior: The current repository uses squared Euclidean distance as its working optimization objective.
- Critical mismatch requiring a separate implementation decision: The thesis objective and the current project objective differ in form and should be treated as a separate implementation decision rather than a simple correction.

Do not modify the fitness function as part of this document.

## 3. Thesis parameter table

The thesis parameter evidence is recorded from page 67:

- population size: 10
- maximum iterations: 100
- reproduction interval: 30
- replacement probability: 0.60
- split probability: 0.10
- merge probability: 0.10
- maximum clusters: 10
- belief-space acceptance ratio: 0.20

These values are thesis evidence and must be treated as page-based evidence only unless the thesis text is read directly for confirmation.

Current reusable defaults are temporary prototype settings and are not thesis-faithful.

## 4. BF lifecycle

Evidence from pages 54-57 describes the BF lifecycle as including:

- chemotaxis
- tumble
- swim
- swarming
- bacterial health
- reproduction
- elimination-dispersal
- retained best solution

The exact loop nesting, coefficients, and stopping rules require additional verification.

This audit records the presence of these elements in the thesis evidence but does not assume the exact procedural details are yet validated in the repository implementation.

## 5. Bacterial interaction

Page 55 contains an attraction-repulsion interaction term Jcc.

This is evidence that bacterial interaction is part of the thesis BF design. The reusable implementation must be inspected for this component.

No coefficient values are inferred here.

## 6. Candidate representation

Pages 60-62 describe bacteria as encoding cluster assignments or cluster-related information and using replacement, split, merge, repair, and gene-transfer operations.

The exact encoding remains ambiguous from the current evidence set alone. The thesis text appears to describe a representation that is more than a simple center matrix.

The current reusable implementation uses center matrices.

This should be recorded as a current implementation choice, not a definitive conclusion that the center representation is wrong.

## 7. Cultural Algorithm

The thesis evidence from pages 58-60 includes the following belief-space components:

- normative knowledge
- situational knowledge
- domain knowledge
- temporal knowledge
- spatial knowledge
- search history

These are thesis-level conceptual components. They should be compared cautiously with the current simplified reusable CBF implementation, which is not yet demonstrated to match the thesis procedure in full detail.

## 8. Structural operations

The thesis evidence includes the following structural operations:

- replacement
- division/splitting
- merging
- repair
- gene transfer
- maximum cluster count

These operations are relevant to the candidate representation and are currently either missing or uncertain in the repository implementation.

The current reusable implementation does not provide evidence that all of these thesis operations are implemented exactly as described.

## 9. Evaluation metrics

The thesis evidence includes the following evaluation elements:

- ordinary Euclidean intra-cluster distance from page 66
- error-rate equation from page 67
- cluster-to-class matching before error calculation
- external labels used only after optimization
- standard deviation across repeated runs

These are thesis-level evaluation requirements and should be treated as evidence statements only until the thesis method is read and matched directly.

The exact repeated-run count and standard-deviation convention require thesis verification.

## 10. Thesis-reported results

The thesis-reported evidence includes Tables 5-3, 5-4, and 5-5.

These tables are thesis-reported evidence and are not current regression targets.

They are not yet reproduced in the repository.

Tests must not be changed merely to force matching with these values.

## 11. Internal inconsistency

The thesis structure contains an internal inconsistency between chapters:

Chapter 5:
- Iris, Wine, Cancer
- K-Means, GA, PSO, BFOA, proposed method

Chapter 6:
- Iris, Wine, Glass, Cancer
- K-Means, GSA, BF, BH, proposed method

This is an internal thesis inconsistency requiring further evidence to resolve.

The current repository should treat this as a documentation mismatch and should not assume one chapter supersedes the other without thesis verification.

## 12. Prioritized decisions

### Priority 1

- choose separate prototype and thesis-compatible objective functions
- verify candidate encoding
- verify Jcc interaction
- verify full BF parameters

### Priority 2

- verify cultural acceptance and influence equations
- verify replacement, split, merge, repair, and transfer
- verify dynamic cluster-number behavior

### Priority 3

- implement external error-rate evaluation
- verify repeated-run count
- verify standard-deviation convention
- resolve comparison-method inconsistency

## 13. Safety boundary

- No algorithm changes are authorized by this document.
- Project A prototype results remain preserved.
- Thesis-compatible repairs must use separate focused commits.
- Project B must not begin until Project A is frozen.

## Conclusion

This audit records the thesis evidence that is currently available without authorizing implementation changes. The repository still contains prototype behavior and temporary settings, but the thesis source evidence shows multiple areas that require separate verification before a thesis-faithful implementation can be claimed.
