"""Numerically stable softmax implementation in NumPy.

Softmax converts a vector of logits into a probability distribution:
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

Why subtract max before exp?
- exp(large_number) overflows to inf.
- softmax(x) == softmax(x + c) for any constant c, so shifting by -max is safe.
- After shifting, the largest value becomes 0 → exp(0) = 1, no overflow.
"""

import numpy as np


def softmax(x):
    """Compute softmax over the last axis.

    Args:
        x: array of shape (..., D) — logits.

    Returns:
        Array of same shape with probabilities summing to 1 along last axis.
    """
    # Shift so max becomes 0 → prevents exp overflow.
    x = x - np.max(x, axis=-1, keepdims=True)

    # Convert shifted logits to unnormalized positive scores.
    exp_x = np.exp(x)

    # Normalize to a probability distribution.
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def batched_softmax(logits):
    """Compute softmax for a batch of logit vectors.

    Args:
        logits: array of shape (B, C) — B samples, C classes.

    Returns:
        Array of shape (B, C) — probability distributions per row.
    """
    # Shift each row independently for numerical stability.
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp_x = np.exp(logits)

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# --- Demo ---
if __name__ == "__main__":
    x = np.array([2, 1, 0])
    print(f"softmax({x}) = {softmax(x)}")
    print(f"sum = {np.sum(softmax(x)):.4f}")

    logits = np.array([[2, 1, 0], [0, 0, 0]])
    print(f"\nbatched_softmax:\n{batched_softmax(logits)}")
