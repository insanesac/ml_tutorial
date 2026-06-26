"""ReLU activation function and its properties.

ReLU(x) = max(0, x)

Why ReLU dominates deep learning:
- Gradient is 1 for positive inputs (no vanishing gradient).
- Gradient is 0 for negative inputs (can cause "dead ReLU").
- Computationally cheap (just a max operation).

This file demonstrates:
1. Basic ReLU behavior on mixed inputs.
2. Vanishing gradient: sigmoid saturates at large values, ReLU does not.
3. Dead ReLU problem: all-negative inputs produce all-zero outputs.
"""

import numpy as np

# --- Basic ReLU ---
z = np.array([-3, -1, 0, 1, 3, 5])

def relu(x):
    """ReLU: max(0, x). Simple, fast, no saturation for positive values."""
    return np.maximum(0, x)

a = relu(z)
print(f"Input:  {z}")
print(f"ReLU:   {a}")

# --- Vanishing gradient intuition ---
# Sigmoid saturates (gradient → 0) for large inputs.
# ReLU does not saturate for positive inputs (gradient = 1).
z_large = np.array([10, 50, 100])

def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x)). Saturates at 0 and 1."""
    return 1 / (1 + np.exp(-x))

print(f"\nSigmoid at large values: {sigmoid(z_large)}")  # all ≈ 1.0 (saturated)
print(f"ReLU at large values:    {relu(z_large)}")      # preserves magnitude

# --- Dead ReLU problem ---
# If all inputs to a ReLU neuron are negative, output is always 0.
# Gradient is also 0 → neuron never updates → permanently "dead".
# Solutions: Leaky ReLU, lower learning rates, proper initialization.
z_all_negative = np.array([-5, -3, -1])
print(f"\nReLU all negative: {relu(z_all_negative)} (all dead)")
