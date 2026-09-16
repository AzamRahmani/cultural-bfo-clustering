import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_loader import load_standardized_dataset


def calculate_manual_inertia(X, labels, centers):
    total = 0.0
    for index, sample in enumerate(X):
        center = centers[labels[index]]
        total += ((sample - center) ** 2).sum()
    return total


X, y, X_scaled, _, _ = load_standardized_dataset("glass")

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

inertia = kmeans.inertia_
manual_inertia = calculate_manual_inertia(
    X_scaled, labels, kmeans.cluster_centers_
)
assert np.isclose(inertia, manual_inertia, rtol=1e-10, atol=1e-10)

silhouette = silhouette_score(X_scaled, labels)

print(f"X.shape: {X.shape}")
print(f"y.shape: {y.shape}")
print(f"Unique labels in y: {len(np.unique(y))}")
print(f"Inertia: {inertia:.12f}")
print(f"Manual inertia: {manual_inertia:.12f}")
print(f"Silhouette score: {silhouette:.12f}")
print(f"Cluster-center shape: {kmeans.cluster_centers_.shape}")