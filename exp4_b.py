import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 2: Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Step 3: Visualize the results

# Plot the original data points
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.title("Original Data")
plt.show()

# Plot the clustered data with centroids
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("Clustered Data with KMeans")
plt.show()

# Step 4: Print the cluster centers
print("Cluster Centers:")
print(centers)

# Step 5: Print the labels of the first 10 data points
print("Labels of the first 10 data points:")
print(y_kmeans[:10])
