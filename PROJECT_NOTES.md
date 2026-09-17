# Thesis Reconstruction Notes

## Verified Dataset Information

### Cancer Dataset

Status: Verified

Dataset:
Breast Cancer Wisconsin (Original)

Source:
OpenML breast-w (data_id=15)

Raw samples: 699

Rows removed because of missing values: 16

Final samples: 683

Features: 9

Classes: 2

Feature names:
- Clump_Thickness
- Cell_Size_Uniformity
- Cell_Shape_Uniformity
- Marginal_Adhesion
- Single_Epi_Cell_Size
- Bare_Nuclei
- Bland_Chromatin
- Normal_Nucleoli
- Mitoses

Rejected Dataset

sklearn.datasets.load_breast_cancer()

Reason:

Produces:
- 569 samples
- 30 features

This does not match the thesis dataset.

## Open Questions

- Verify exact thesis preprocessing.
- Verify missing-value handling from the thesis.
- Verify thesis Cancer benchmark results.

## Literature Review

### Critical Review of Bio-Inspired Algorithms

Main lessons:
- Novel biological terminology does not automatically mean algorithmic novelty.
- Claims require strong baselines and repeated experiments.
- Scalability, convergence, reliability, and interpretability must be evaluated.
- Improvements should be separated from reproduction.