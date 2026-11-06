# Project B Data Record

Project B (sensor data, learned representations, anomaly detection, edge
feasibility) is separate from the frozen Project A (`project-a-v1`).

## Dataset: UCI Human Activity Recognition Using Smartphones

| Item | Value |
|---|---|
| Source | UCI Machine Learning Repository, dataset ID 240 |
| DOI | 10.24432/C54S4K |
| License | CC BY 4.0 |
| Download URL | https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip |
| Downloaded | 2026-11-06 |
| Local path (not in Git) | `data/raw/uci_har_240.zip`, extracted to `data/uci_har/` |
| Outer zip size | 61,005,872 bytes |
| Outer zip SHA-256 | `C00B803081A5C797CD5E4B83700A9810B38D53D9D84E01917E090E1FDBC81031` |
| Inner `UCI HAR Dataset.zip` SHA-256 | `2045E435C955214B38145FB5FA00776C72814F01B203FEC405152DAC7D5BFEB0` |

Citation: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., and Parra, X.
(2013). Human Activity Recognition Using Smartphones. UCI Machine Learning
Repository. https://doi.org/10.24432/C54S4K. Introductory paper: "A Public
Domain Dataset for Human Activity Recognition using Smartphones", ESANN 2013.

## Structural check (2026-11-06)

Only file structure was checked; no statistics of the data were computed
before the Project B protocol is written.

- 561 features listed in `features.txt`; 6 activities: WALKING,
  WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
- Train: 7,352 windows x 561 features, 21 subjects.
- Test: 2,947 windows x 561 features, 9 subjects.
- Total: 10,299 windows, 30 subjects; the provider's split is by subject.
- Nine raw inertial channels per split (body and total acceleration x/y/z,
  body gyroscope x/y/z), each 128 samples per window.
- No missing or infinite values; all label files contain labels 1 to 6.
- The inner archive also contains macOS metadata (`__MACOSX`, `.DS_Store`),
  which is ignored.
