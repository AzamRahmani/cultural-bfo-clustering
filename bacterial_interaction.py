import numpy as np


def bacterial_interaction(
    candidate,
    population,
    attraction_depth,
    attraction_width,
    repulsion_height,
    repulsion_width,
    include_self=True,
    self_index=None,
):
    """Calculate the attraction-repulsion Jcc for a candidate bacterium."""
    candidate = np.asarray(candidate, dtype=float)
    population = np.asarray(population, dtype=float)

    if candidate.ndim != 1:
        raise ValueError(
            "candidate must have shape (number_of_dimensions,)."
        )

    if population.ndim != 2:
        raise ValueError(
            "population must have shape "
            "(number_of_bacteria, number_of_dimensions)."
        )

    if population.shape[1] != candidate.shape[0]:
        raise ValueError(
            "candidate and population must have the same number "
            "of dimensions."
        )

    if not include_self:
        if self_index is None:
            raise ValueError(
                "self_index is required when include_self is False."
            )

        if (
            isinstance(self_index, (bool, np.bool_))
            or not isinstance(self_index, (int, np.integer))
        ):
            raise ValueError("self_index must be an integer row index.")

        if not 0 <= self_index < population.shape[0]:
            raise ValueError(
                "self_index must identify a row in population."
            )

    squared_distances = np.sum(
        (candidate[np.newaxis, :] - population) ** 2,
        axis=1,
    )
    attraction = -attraction_depth * np.exp(
        -attraction_width * squared_distances
    )
    repulsion = repulsion_height * np.exp(
        -repulsion_width * squared_distances
    )
    interactions = attraction + repulsion

    if not include_self:
        interactions = np.delete(interactions, self_index)

    return float(np.sum(interactions))