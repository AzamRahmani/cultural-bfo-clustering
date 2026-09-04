# Wine Dataset Baseline Clustering Experiment
# This script demonstrates basic K-Means clustering on the Wine dataset

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
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

pca = PCA(
    n_components=2,
)

X_visual = pca.fit_transform(X_scaled)

assert X_visual.shape == (178, 2)
assert 0 < pca.explained_variance_ratio_.sum() <= 1

fig, ax = plt.subplots(
    figsize=(8, 6),
)

colors = ["#4C78A8", "#F58518", "#54A24B"]

for cluster_label in range(3):
    cluster_mask = kmeans_labels == cluster_label

    ax.scatter(
        X_visual[cluster_mask, 0],
        X_visual[cluster_mask, 1],
        label=f"Cluster {cluster_label}",
        color=colors[cluster_label],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3,
    )

ax.set_title(
    "Wine K-Means Clusters Visualized with PCA"
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
ax.grid(alpha=0.25)

ax.text(
    0.02,
    0.02,
    "PCA is used only for visualization.\nK-Means was trained on all 13 standardized features.",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
)

plt.tight_layout()

output_file = "wine_clusters.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Chart saved to: {output_file}")
print(
    "PCA explained variance for two components: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)
