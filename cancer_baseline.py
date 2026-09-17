import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_loader import load_standardized_dataset


def calculate_manual_inertia(X, labels, centers):
    total_distance = 0.0

    for index, sample in enumerate(X):
        cluster_center = centers[labels[index]]
        total_distance += np.sum(
            (sample - cluster_center) ** 2
        )

    return total_distance


X, y, X_scaled, _, _ = load_standardized_dataset("cancer")

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10,
)
labels = kmeans.fit_predict(X_scaled)

inertia = kmeans.inertia_
manual_inertia = calculate_manual_inertia(
    X_scaled,
    labels,
    kmeans.cluster_centers_,
)
silhouette = silhouette_score(X_scaled, labels)

assert X.shape == (683, 9)
assert y.shape == (683,)
assert len(np.unique(labels)) == 2
assert kmeans.cluster_centers_.shape == (2, 9)
assert np.isclose(manual_inertia, inertia)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"unique label count: {len(np.unique(labels))}")
print(f"inertia: {inertia:.4f}")
print(f"manual inertia: {manual_inertia:.4f}")
print(f"silhouette score: {silhouette:.4f}")
print(f"cluster-center shape: {kmeans.cluster_centers_.shape}")
print("manual inertia matches sklearn inertia: True")
