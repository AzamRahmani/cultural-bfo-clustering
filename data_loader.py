from sklearn.datasets import fetch_openml, load_iris, load_wine
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
    elif normalized_name == "glass":
        dataset = fetch_openml(
            name="glass",
            version=1,
            as_frame=False,
            parser="liac-arff",
        )
    else:
        raise ValueError(
            "Unsupported dataset. Choose 'iris', 'wine', or 'glass'."
        )

    X = dataset.data
    y = dataset.target
    feature_names = list(dataset.feature_names)
    if normalized_name == "glass":
        class_names = sorted(set(y.tolist()))
    else:
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
        "glass": ((214, 9), (214,), 9),
    }

    for dataset_name in ["iris", "wine", "glass"]:
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
        expected_class_count = 6 if dataset_name == "glass" else 3
        assert len(class_names) == expected_class_count
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
