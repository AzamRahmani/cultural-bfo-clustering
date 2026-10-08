import math
import numbers

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix


def _validated_identifiers(values, name):
    try:
        identifiers = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a one-dimensional array of discrete identifiers."
        ) from error

    if identifiers.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    for identifier in identifiers:
        if isinstance(identifier, np.generic):
            identifier = identifier.item()

        if identifier is None:
            raise ValueError(
                f"{name} must not contain missing or non-finite values."
            )

        if isinstance(identifier, numbers.Real):
            try:
                is_finite = math.isfinite(identifier)
            except (OverflowError, TypeError, ValueError):
                is_finite = False
            if not is_finite:
                raise ValueError(
                    f"{name} must not contain missing or non-finite values."
                )
        elif not isinstance(identifier, (str, bytes)):
            raise ValueError(
                f"{name} must contain discrete identifiers."
            )

        try:
            hash(identifier)
        except TypeError as error:
            raise ValueError(
                f"{name} must contain discrete identifiers."
            ) from error

    return identifiers


def clustering_error_rate(y_true, cluster_labels):
    """Evaluate clustering labels using an optimal one-to-one class mapping."""
    y_true = _validated_identifiers(y_true, "y_true")
    cluster_labels = _validated_identifiers(
        cluster_labels,
        "cluster_labels",
    )

    if y_true.shape[0] != cluster_labels.shape[0]:
        raise ValueError(
            "y_true and cluster_labels must contain the same number of samples."
        )
    if y_true.shape[0] == 0:
        raise ValueError("At least one sample is required.")

    try:
        class_identifiers = np.unique(y_true)
        cluster_identifiers = np.unique(cluster_labels)
    except TypeError as error:
        raise ValueError(
            "Class and cluster labels must be mutually comparable discrete identifiers."
        ) from error

    if class_identifiers.size != cluster_identifiers.size:
        raise ValueError(
            "The number of predicted clusters must equal the number of known classes."
        )

    try:
        counts = contingency_matrix(
            y_true,
            cluster_labels,
            sparse=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Class and cluster labels must be mutually comparable discrete identifiers."
        ) from error

    class_indices, cluster_indices = linear_sum_assignment(-counts)
    mapping = {
        cluster_identifiers[cluster_index].item()
        if isinstance(cluster_identifiers[cluster_index], np.generic)
        else cluster_identifiers[cluster_index]:
        class_identifiers[class_index].item()
        if isinstance(class_identifiers[class_index], np.generic)
        else class_identifiers[class_index]
        for class_index, cluster_index in zip(
            class_indices,
            cluster_indices,
        )
    }

    correct_count = int(counts[class_indices, cluster_indices].sum())
    incorrect_count = int(y_true.shape[0] - correct_count)
    accuracy_percent = 100.0 * correct_count / y_true.shape[0]

    return {
        "error_rate_percent": 100.0 - accuracy_percent,
        "accuracy_percent": accuracy_percent,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "mapping": mapping,
        "contingency_matrix": counts,
    }