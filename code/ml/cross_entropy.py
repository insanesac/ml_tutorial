"""Cross-entropy loss demonstration.

Cross-entropy measures the difference between predicted and true distributions:
    CE = -mean(log(p[correct_class]))

Key properties:
- Penalizes confident wrong predictions heavily (log(small) → large negative).
- Minimum value is 0 (perfect prediction: p[correct] = 1, log(1) = 0).
- No upper bound — can be very large when model is confidently wrong.

This file demonstrates:
1. Correct predictions with high confidence → low loss.
2. Confidently wrong predictions → high loss.
"""

import numpy as np

# --- Softmax probabilities for 3 classes, 6 samples ---
# Logits are designed so model predicts the correct class with high confidence.
logits = np.array([
    [2.0, 0.5, 0.1],  # class 0 (highest logit)
    [2.2, 0.4, 0.2],  # class 0
    [0.2, 2.0, 0.5],  # class 1
    [0.3, 2.2, 0.4],  # class 1
    [0.1, 0.5, 2.0],  # class 2
    [0.2, 0.4, 2.2]   # class 2
])  # shape (6, 3)

y_true = np.array([0, 0, 1, 1, 2, 2])  # shape (6,) — correct class indices

# --- Softmax ---
def softmax(z):
    """Numerically stable softmax over the last axis."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- Case 1: Model predicts correctly ---
probs = softmax(logits)  # shape (6, 3)
m = len(y_true)

# CE loss: -mean(log(probability of correct class))
loss = -np.mean(np.log(probs[range(m), y_true]))
print(f"Cross-entropy loss (correct): {loss:.4f}")

# --- Case 2: Model is confidently wrong ---
# Model predicts class 2 for sample 0 (true: 0)
# Model predicts class 0 for sample 1 (true: 1)
logits_wrong = np.array([
    [0.1, 0.5, 2.0],  # model thinks class 2, true is 0
    [2.0, 0.5, 0.1],  # model thinks class 0, true is 1
])
y_wrong = np.array([0, 1])

probs_wrong = softmax(logits_wrong)
loss_wrong = -np.mean(np.log(probs_wrong[range(len(y_wrong)), y_wrong]))
print(f"Confidently wrong loss:        {loss_wrong:.4f}")
# Loss is much higher because model assigns very low probability to correct class.
