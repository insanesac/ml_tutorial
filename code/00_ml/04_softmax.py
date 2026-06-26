"""Softmax regression (multiclass logistic regression) with gradient descent.

Model: p = softmax(X @ W + b)
Loss:  Cross-entropy = -(1/m) * sum(log(p[correct_class]))
Gradient w.r.t. W: dW = (1/m) * X^T @ (p - y_onehot)
Gradient w.r.t. b: db = (1/m) * sum(p - y_onehot, axis=0)

Key gradient insight: dL/dz = p - y_onehot
- For correct class: subtract 1 from its probability.
- For wrong classes: no change.
- This is the same (p - y) pattern as binary logistic regression.

Dataset: 3 classes — small → class 0, medium → class 1, large → class 2.
"""

import numpy as np

# --- Dataset ---
X = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10],
    [20, 20],
    [21, 20]
])  # shape (6, 2) — 6 samples, 2 features

y = np.array([0, 0, 1, 1, 2, 2])  # shape (6,) — class labels

# --- Initialize parameters ---
# W: shape (2, 3) — 2 features, 3 classes
W = np.random.randn(X.shape[1], 3) * 0.01
# b: shape (3,) — one bias per class
b = np.zeros(3)

# --- Softmax function ---
def softmax(z):
    """Numerically stable softmax over the last axis.

    Args:
        z: logits, shape (B, C)

    Returns:
        Probabilities, shape (B, C), each row sums to 1.
    """
    # Subtract max for numerical stability (prevents exp overflow).
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- Hyperparameters ---
epochs = 1000
lr = 0.1

# --- Training loop ---
for i in range(epochs):
    # Forward pass: (6, 2) @ (2, 3) + (3,) → (6, 3)
    z = X @ W + b
    p = softmax(z)  # shape (6, 3)

    # Cross-entropy loss: -mean(log(p[correct_class]))
    m = X.shape[0]
    log_likelihood = -np.log(p[range(m), y])  # shape (6,)
    loss = np.sum(log_likelihood) / m
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    # Gradient: dz = p - y_onehot
    # For correct class: dz[i, y[i]] -= 1
    # For wrong classes: dz stays as p[i, j]
    dz = p.copy()           # shape (6, 3)
    dz[range(m), y] -= 1    # subtract 1 from correct class probability
    dz /= m                 # average over batch

    # dW = X^T @ dz → (2, 6) @ (6, 3) → (2, 3)
    dW = X.T @ dz
    # db = sum(dz, axis=0) → (3,)
    db = np.sum(dz, axis=0)

    # Gradient descent update.
    W -= lr * dW
    b -= lr * db

print(f"\nFinal predictions: {np.argmax(softmax(X @ W + b), axis=1)}")
