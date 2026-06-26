"""Neural network forward pass (inference only, no training).

Architecture: 2-layer MLP for 3-class classification.
    Input (2) → Linear → ReLU → Linear → Softmax → Output (3)

Forward pass:
    z1 = X @ W1 + b1   (linear layer 1)
    a1 = relu(z1)       (activation)
    z2 = a1 @ W2 + b2   (linear layer 2)
    a2 = softmax(z2)    (output probabilities)

This file shows only the forward pass. See backpropagation.py for training.

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
])  # shape (6, 2) — 6 samples, 2 features

y = np.array([0, 0, 1, 1, 2, 2])  # shape (6,) — class labels

# --- Network architecture ---
input_dim = 2
hidden_dim = 4
output_dim = 3

# --- Initialize weights (small random values) ---
# W1: (2, 4) — input to hidden
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)  # (4,)
# W2: (4, 3) — hidden to output
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)  # (3,)

# --- Activation functions ---
def relu(z):
    """ReLU: max(0, z). Simple, no saturation for positive values."""
    return np.maximum(0, z)

def softmax(z):
    """Numerically stable softmax over the last axis."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- Forward pass ---
# Layer 1: (6, 2) @ (2, 4) + (4,) → (6, 4)
z1 = X @ W1 + b1  # pre-activation
a1 = relu(z1)     # activation: (6, 4)

# Layer 2: (6, 4) @ (4, 3) + (3,) → (6, 3)
z2 = a1 @ W2 + b2  # pre-activation (logits)
a2 = softmax(z2)    # output probabilities: (6, 3)

print(f"Output shape: {a2.shape}")       # (6, 3)
print(f"Predictions: {np.argmax(a2, axis=1)}")  # argmax over classes
print(f"True labels: {y}")
