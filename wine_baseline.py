# Wine Dataset Baseline Clustering Experiment
# This script demonstrates basic K-Means clustering on the Wine dataset

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def calculate_within_cluster_distance(X, labels, centers):
    """
    Calculate the sum of squared Euclidean distances
    between every sample and its assigned cluster center.
    """
    total_distance = 0.0

    for index, sample in enumerate(X):
        cluster_label = labels[index]
        cluster_center = centers[cluster_label]
        squared_distance = ((sample - cluster_center) ** 2).sum()
        total_distance += squared_distance

    return total_distance


wine = load_wine()
X = wine.data
y = wine.target

print("Wine Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of known classes: {len(wine.target_names)}")
print(f"Feature names: {list(wine.feature_names)}")
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

calculated_distance = calculate_within_cluster_distance(
    X_scaled, kmeans_labels, kmeans.cluster_centers_
)

inertia = kmeans.inertia_
difference = abs(calculated_distance - inertia)
silhouette = silhouette_score(X_scaled, kmeans_labels)

print("Wine K-Means Results:")
print(f"K-Means inertia: {inertia:.4f}")
print(f"Calculated within-cluster distance: {calculated_distance:.4f}")
print(f"Difference from K-Means inertia: {difference:.16e}")
print(f"Silhouette Score: {silhouette:.4f}")
