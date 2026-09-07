from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler


def load_standardized_dataset(dataset_name):
    """
    Load a supported dataset and standardize its feature values.

    Returns:
        X: original feature data
        y: known reference labels
        X_scaled: standardized feature data
        feature_names: names of the input features
        class_names: names of the known classes
    """
    normalized_name = dataset_name.strip().lower()

    if normalized_name == "iris":
        dataset = load_iris()
    elif normalized_name == "wine":
        dataset = load_wine()
    else:
        raise ValueError(
            "Unsupported dataset. Choose 'iris' or 'wine'."
        )

    X = dataset.data
    y = dataset.target
    feature_names = list(dataset.feature_names)
    class_names = list(dataset.target_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return (
        X,
        y,
        X_scaled,
        feature_names,
        class_names,
    )


if __name__ == "__main__":
    expected_shapes = {
        "iris": ((150, 4), (150,), 4),
        "wine": ((178, 13), (178,), 13),
    }

    for dataset_name in ["iris", "wine"]:
        X, y, X_scaled, feature_names, class_names = (
            load_standardized_dataset(dataset_name)
        )
        expected_X_shape, expected_y_shape, expected_feature_count = (
            expected_shapes[dataset_name]
        )

        assert X.shape == expected_X_shape
        assert y.shape == expected_y_shape
        assert X_scaled.shape == expected_X_shape
        assert len(feature_names) == expected_feature_count
        assert len(class_names) == 3
        assert abs(X_scaled.mean()) < 1e-10

        print(f"Dataset: {dataset_name.capitalize()}")
        print(f"Original shape: {X.shape}")
        print(f"Standardized shape: {X_scaled.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Features: {len(feature_names)}")
        print(f"Known classes: {len(class_names)}")

    try:
        load_standardized_dataset("unknown")
    except ValueError as error:
        print(f"Expected error: {error}")
    else:
        raise AssertionError(
            "Unsupported dataset did not raise ValueError."
        )

    print("Dataset loader checks passed.")
