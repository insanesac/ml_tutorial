"""K-Means clustering.

Algorithm (Lloyd's algorithm):
1. Initialize k centroids randomly from the data points.
2. Assignment step: assign each point to nearest centroid.
3. Update step: recompute centroids as mean of assigned points.
4. Repeat 2-3 until centroids converge (stop moving).

Convergence: K-Means always converges because each step reduces
the within-cluster sum of squares (WCSS). However, it may converge
to a local minimum — sensitive to initialization.

Dataset: 3 well-separated clusters at ~(1,1), ~(10,10), ~(20,20).
Expected centroids: [~1.33, ~1.33], [~10.33, ~10.33], [~20.33, ~20.33]
"""

import numpy as np

# --- Dataset ---
X = np.array([
    [1, 1], [1, 2], [2, 1],       # cluster 0
    [10, 10], [10, 11], [11, 10], # cluster 1
    [20, 20], [20, 21], [21, 20]  # cluster 2
])  # shape (9, 2)

k = 3

# --- Initialize centroids randomly from data points ---
np.random.seed(42)  # for reproducibility
centroids = X[np.random.choice(X.shape[0], k, replace=False)]  # shape (3, 2)

max_iter = 100

# --- K-Means loop ---
for _ in range(max_iter):
    old_centroids = centroids.copy()
    clusters = [[] for _ in range(k)]

    # Assignment step: assign each point to nearest centroid.
    for x in X:
        # Distance from x to each centroid.
        distances = [np.linalg.norm(c - x) for c in centroids]
        closest = np.argmin(distances)  # index of nearest centroid
        clusters[closest].append(x)

    # Update step: recompute centroids as mean of assigned points.
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:  # guard against empty clusters
            centroids[i] = np.mean(cluster, axis=0)

    # Convergence check: stop if centroids didn't move.
    if np.allclose(old_centroids, centroids):
        break

print(f"Centroids:\n{centroids}")
