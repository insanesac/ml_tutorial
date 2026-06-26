import numpy as np

X = np.array([
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 1],
    [6, 1]
])
y = np.array([0, 0, 0, 1, 1, 1])

# Shuffle and split
np.random.seed(42)
indices = np.random.permutation(len(X))

# 60% train, 20% val, 20% test
n = len(X)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Train: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")

print(f"\nWhy not tune on test data?")
print(f"Tuning on test data leaks information about the test set into the model.")
print(f"The model becomes overfit to the test set, and true generalization is unknown.")
