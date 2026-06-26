"""K-Nearest Neighbors (KNN) classification.

KNN is a lazy learner — no training step.
- Training: just store the data.
- Inference: find k nearest neighbors by distance, majority vote.

Why training is fast: nothing to compute.
Why inference is slow: must compute distance to every training point.

Dataset: two clusters — class 0 near (1,1), class 1 near (10,10).
Expected: x_new = (1.5, 1.5) → class 0.
"""

import numpy as np

# --- Training data (just stored, no training step) ---
X_train = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10]
])  # shape (4, 2)

y_train = np.array([0, 0, 1, 1])  # shape (4,)

# --- Query point ---
x_new = np.array([1.5, 1.5])  # shape (2,)

k = 3

# --- Inference ---
# Compute Euclidean distance from x_new to every training point.
# Broadcasting: (4, 2) - (2,) → (4, 2), then norm over axis=1 → (4,)
distances = np.linalg.norm(X_train - x_new, axis=1)  # shape (4,)

# Get indices of k nearest neighbors (sorted by distance).
dist_index = np.argsort(distances)  # shape (4,) — sorted indices
k_index = dist_index[:k]             # shape (k,) — top k indices

# Get labels of the k nearest neighbors.
neighbor_labels = y_train[k_index]  # shape (k,)

# Majority vote: bincount counts occurrences of each class label.
# argmax returns the label with the most votes.
prediction = np.argmax(np.bincount(neighbor_labels))

print(prediction)  # Expected: 0