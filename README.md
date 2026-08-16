# Cultural Bacterial Foraging for Data Clustering

This project is a Python reimplementation and evaluation of a master's thesis combining Bacterial Foraging Optimization with a Cultural Algorithm for data clustering.

## Current Status

**Phase 1 - K-Means baseline on the Iris dataset**

## Current Features

- Loads the Iris dataset (150 samples, 4 features, 3 classes)
- Standardizes four input features using StandardScaler
- Runs K-Means clustering with three clusters
- Reports inertia (within-cluster compactness)
- Reports silhouette score (cluster separation quality)
- Uses a fixed random seed (42) for reproducibility

## Verified Thesis Scope

**Datasets:**
- Iris (150 samples, 4 features, 3 clusters)
- Wine (178 samples, 13 features, 3 clusters)
- Glass (214 samples, 9 features, 6 clusters)
- Cancer (683 samples, 9 features, 2 clusters)

**Comparison Methods:**
- K-Means
- GSA (Gravitational Search Algorithm)
- BF (Bacterial Foraging)
- BH (Black-Hole)
- CBF (Cultural Bacterial Foraging) - proposed hybrid method

**Evaluation Metrics:**
- Within-cluster distance
- Standard deviation
- Error rate

## Planned Implementation Order

1. K-Means baseline ✓
2. Exact thesis metric implementation
3. Basic Bacterial Foraging algorithm
4. Cultural belief space
5. Hybrid CBF (Cultural Bacterial Foraging)
6. Four-dataset comparison with all methods

## Installation (Windows PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy scikit-learn matplotlib
```

## Running the Baseline

```powershell
python iris_baseline.py
```

### Example Output

```
Dataset Information:
Number of samples: 150
Number of features: 4
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Number of known classes: 3

K-Means Clustering Results:
K-Means inertia: 139.8205
Silhouette Score: 0.4599
```

## Important Limitation

The current inertia result (139.8205) is from scikit-learn's K-Means algorithm. This value must not be claimed as identical to the thesis within-cluster-distance result until:

1. The exact thesis formula is reproduced
2. The thesis data preprocessing is replicated exactly
3. Comparison is performed under identical conditions

## Author

Azam Rahmani
