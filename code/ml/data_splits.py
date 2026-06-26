"""Train / Validation / Test data splits.

Three-way split:
- Train: used to fit model parameters (weights, biases).
- Validation: used to tune hyperparameters (learning rate, epochs, etc.).
- Test: used only once at the end to estimate generalization performance.

Why not tune on test data?
- Tuning on test leaks information about the test set into the model.
- The model becomes overfit to the test set.
- True generalization performance becomes unknown.
- This is called "data leakage" or "test set contamination."

Split ratios (common):
- 60/20/20 — small datasets
- 80/10/10 — medium datasets
- 98/1/1   — very large datasets
"""

import numpy as np

# --- Dataset ---
X = np.array([
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 1]
])  # shape (6, 2)

y = np.array([0, 0, 0, 1, 1, 1])  # shape (6,)

# --- Shuffle indices for random split ---
np.random.seed(42)  # for reproducibility
indices = np.random.permutation(len(X))  # shape (6,) — shuffled indices

# --- Split: 60% train, 20% val, 20% test ---
n = len(X)
train_end = int(0.6 * n)  # 3
val_end = int(0.8 * n)    # 4

train_idx = indices[:train_end]      # first 60%
val_idx = indices[train_end:val_end] # next 20%
test_idx = indices[val_end:]         # last 20%

# Index into data to create splits.
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")

print(f"\nWhy not tune on test data?")
print(f"Tuning on test data leaks information about the test set into the model.")
print(f"The model becomes overfit to the test set, and true generalization is unknown.")
