"""Multi-Feature Logistic Regression with Gradient Descent.

Model: p = sigmoid(X @ w + b)
Loss:  BCE = -mean(y * log(p) + (1-y) * log(1-p))
Gradient w.r.t. w: dw = (1/m) * X^T @ (p - y)
Gradient w.r.t. b: db = (1/m) * sum(p - y)

Note: gradient of sigmoid + BCE simplifies to (p - y),
which is clean and avoids the vanishing gradient of sigmoid alone.

Dataset: both features increase with class label.
Expected: w1 > 0, w2 > 0, b < 0
"""

import numpy as np

# --- Dataset ---
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 3],
    [4, 4]
])  # shape (6, 2)

y = np.array([0, 0, 0, 1, 1, 1])  # shape (6,)

# --- Initialize parameters ---
w = np.zeros(X.shape[1])  # shape (2,)
b = 0.0

# --- Hyperparameters ---
epochs = 1000
lr = 0.1
eps = 1e-15  # clipping constant to prevent log(0)

# --- Training loop ---
for i in range(epochs):
    # Linear combination: (6, 2) @ (2,) + scalar → (6,)
    z = X @ w + b

    # Sigmoid: squashes to (0, 1) probability.
    p = 1 / (1 + np.exp(-z))  # shape (6,)

    # Clip to avoid log(0) → -inf.
    p = np.clip(p, eps, 1 - eps)

    # Binary cross-entropy loss.
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    # Gradients (gradient of BCE w.r.t. sigmoid output is (p - y)).
    # dw = (1/m) * X^T @ (p - y) → (2, 6) @ (6,) → (2,)
    dw = X.T @ (p - y) / len(X)
    # db = (1/m) * sum(p - y)    → scalar
    db = np.mean(p - y)

    # Gradient descent update.
    w -= lr * dw
    b -= lr * db

print(f"\nFinal w: {w}")
print(f"Final b: {b:.4f}")
print(f"\nPredictions: {p}")
