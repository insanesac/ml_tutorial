"""Neural network with backpropagation training.

Architecture: 2-layer MLP for 3-class classification.
    Input (2) → Linear → ReLU → Linear → Softmax → Output (3)

Forward pass:
    z1 = X @ W1 + b1   → (6, 4)
    a1 = relu(z1)       → (6, 4)
    z2 = a1 @ W2 + b2   → (6, 3)
    a2 = softmax(z2)    → (6, 3)

Backpropagation (chain rule, output → input):
    dz2 = a2 - y_onehot           (gradient of softmax + CE)
    dW2 = a1^T @ dz2              (gradient for W2)
    da1 = dz2 @ W2^T              (propagate to hidden layer)
    dz1 = da1 * relu'(z1)         (backprop through ReLU)
    dW1 = X^T @ dz1               (gradient for W1)

Key insight: gradient of softmax + cross-entropy w.r.t. logits = (p - y).
ReLU gradient: 1 if z > 0, else 0.

Dataset: 3 classes — small → 0, medium → 1, large → 2.
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
])  # shape (6, 2)

y = np.array([0, 0, 1, 1, 2, 2])  # shape (6,)

# --- Network architecture ---
input_dim = 2
hidden_dim = 4
output_dim = 3

# --- Initialize weights (small random values) ---
W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # (2, 4)
b1 = np.zeros(hidden_dim)                           # (4,)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01  # (4, 3)
b2 = np.zeros(output_dim)                            # (3,)

# --- Activation functions ---
def relu(z):
    """ReLU: max(0, z). Gradient: 1 if z > 0, else 0."""
    return np.maximum(0, z)

def softmax(z):
    """Numerically stable softmax over the last axis."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- Hyperparameters ---
epochs = 1000
lr = 0.1

# --- Training loop ---
for i in range(epochs):
    # === Forward pass ===
    z1 = X @ W1 + b1  # (6, 2) @ (2, 4) → (6, 4)
    a1 = relu(z1)      # (6, 4)
    z2 = a1 @ W2 + b2  # (6, 4) @ (4, 3) → (6, 3)
    a2 = softmax(z2)    # (6, 3) — predicted probabilities

    # === Loss: cross-entropy ===
    m = X.shape[0]
    log_likelihood = -np.log(a2[range(m), y])  # (6,)
    loss = np.sum(log_likelihood) / m
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    # === Backpropagation ===
    # Output layer: dz2 = a2 - y_onehot (gradient of softmax + CE w.r.t. logits)
    dz2 = a2.copy()          # (6, 3)
    dz2[range(m), y] -= 1    # subtract 1 from correct class
    dz2 /= m                 # average over batch

    # Gradients for W2, b2.
    # dW2 = a1^T @ dz2 → (4, 6) @ (6, 3) → (4, 3)
    dW2 = a1.T @ dz2
    # db2 = sum(dz2, axis=0) → (3,)
    db2 = np.sum(dz2, axis=0)

    # Propagate to hidden layer.
    # da1 = dz2 @ W2^T → (6, 3) @ (3, 4) → (6, 4)
    da1 = dz2 @ W2.T
    # Backprop through ReLU: gradient is 1 where z1 > 0, else 0.
    dz1 = da1 * (z1 > 0).astype(float)  # (6, 4)

    # Gradients for W1, b1.
    # dW1 = X^T @ dz1 → (2, 6) @ (6, 4) → (2, 4)
    dW1 = X.T @ dz1
    # db1 = sum(dz1, axis=0) → (4,)
    db1 = np.sum(dz1, axis=0)

    # === Gradient descent update ===
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print(f"\nFinal predictions: {np.argmax(softmax(relu(X @ W1 + b1) @ W2 + b2), axis=1)}")
