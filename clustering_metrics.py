import numpy as np


def _validated_metric_inputs(X, labels, centers):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centers = np.asarray(centers)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")
    if centers.ndim != 2:
        raise ValueError("centers must be a two-dimensional array.")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must contain the same number of samples.")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers must contain the same number of features.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer cluster identifiers.")

    for name, values in (("X", X), ("centers", centers)):
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise ValueError(f"{name} must contain finite real numeric values.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite real numeric values.")

    if np.any(labels < 0) or np.any(labels >= centers.shape[0]):
        raise ValueError("labels must reference valid center indices.")

    try:
        X = X.astype(np.float64, copy=False)
        centers = centers.astype(np.float64, copy=False)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("X and centers must contain finite real numeric values.") from error

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(centers)):
        raise ValueError("X and centers must contain finite real numeric values.")

    return X, labels, centers


def within_cluster_sum_squared_distance(X, labels, centers):
    """Return summed squared distances for X (n_samples, n_features),
    labels (n_samples,), and centers (n_clusters, n_features).

    This is the current prototype objective: square coordinate differences
    to each sample's assigned center and sum them. Adding this metric does
    not establish thesis reproduction.
    """
    X, labels, centers = _validated_metric_inputs(X, labels, centers)
    differences = X - centers[labels]
    return float(np.sum(differences**2))


def within_cluster_sum_euclidean_distance(X, labels, centers):
    """Return summed Euclidean distances for X (n_samples, n_features),
    labels (n_samples,), and centers (n_clusters, n_features).

    This metric sums each sample's Euclidean distance to its assigned center
    and is added for thesis-compatible evaluation. Adding it does not
    establish thesis reproduction.
    """
    X, labels, centers = _validated_metric_inputs(X, labels, centers)
    differences = X - centers[labels]
    return float(np.sum(np.linalg.norm(differences, axis=1)))