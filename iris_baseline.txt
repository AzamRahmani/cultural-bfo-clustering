# Iris Dataset Baseline Clustering Experiment
# This script demonstrates basic K-Means clustering on the Iris dataset

# Import required libraries
from sklearn.datasets import load_iris
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


# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature data (measurements)
y = iris.target  # True labels (species) - used only for evaluation, not training

# Print dataset information
print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(iris.feature_names)}")
print(f"Number of known classes: {len(iris.target_names)}")
print()

# Standardize the features
# This ensures all features are on the same scale (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform K-Means clustering
# n_clusters=3: cluster into 3 groups (matching the 3 iris species)
# random_state=42: ensures reproducible results
# n_init=10: runs the algorithm 10 times and returns the best result
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)  # Train on X_scaled, NOT on y

calculated_distance = calculate_within_cluster_distance(
    X_scaled, kmeans_labels, kmeans.cluster_centers_
)

# Calculate clustering quality metrics
# Inertia: sum of squared distances from points to their assigned cluster center
# Lower inertia indicates more compact clusters
inertia = kmeans.inertia_
difference = abs(calculated_distance - inertia)

# Silhouette Score: measures how similar a point is to its own cluster vs other clusters
# Range: -1 to 1, where higher is better
silhouette = silhouette_score(X_scaled, kmeans_labels)

# Print results
print("K-Means Clustering Results:")
print(f"Inertia (within-cluster distance): {inertia:.4f}")
print(f"Calculated within-cluster distance: {calculated_distance:.4f}")
print(f"Difference from K-Means inertia: {difference:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")
