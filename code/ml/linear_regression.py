"""Multi-Feature Linear Regression with Gradient Descent.

Model: y_pred = X @ w + b
Loss:  MSE = (1/m) * sum((y_pred - y)^2)
Gradient w.r.t. w: dw = (1/m) * X^T @ (y_pred - y)
Gradient w.r.t. b: db = (1/m) * sum(y_pred - y)

Dataset: y = 2*x1 + 3*x2 + 1
Expected: w ≈ [2, 3], b ≈ 1
"""

import numpy as np

# --- Dataset ---
# True relationship: y = 2*x1 + 3*x2 + 1
X = np.array([
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 2],
    [5, 4]
])  # shape (5, 2) — 5 samples, 2 features

y = np.array([9, 8, 16, 15, 23])  # shape (5,) — target values

# --- Initialize parameters ---
w = np.zeros(X.shape[1])  # shape (2,) — one weight per feature
b = 0.0                   # scalar bias

# --- Hyperparameters ---
epochs = 1000
lr = 0.01

# --- Training loop ---
for i in range(epochs):
    # Forward pass: (5, 2) @ (2,) + scalar → (5,)
    y_pred = X @ w + b

    # Error (residuals).
    error = y_pred - y  # shape (5,)

    # MSE loss.
    loss = np.mean(error ** 2)
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    # Gradients.
    # dw = (1/m) * X^T @ error → (2, 5) @ (5,) → (2,)
    dw = X.T @ error / len(X)
    # db = (1/m) * sum(error)  → scalar
    db = np.mean(error)

    # Gradient descent update.
    w -= lr * dw
    b -= lr * db

print(f"\nFinal w: {w}")
print(f"Final b: {b:.4f}")
