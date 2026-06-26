"""Cross-entropy loss in NumPy.

Cross-entropy measures the difference between two probability distributions:
    CE(y_true, y_pred) = -sum(y_true * log(y_pred))

For sparse labels (class indices, not one-hot):
    CE = -log(y_pred[correct_class])

Why cross-entropy?
- It's the standard loss for classification.
- Gradient w.r.t. logits is simply (p - y), which is clean and stable.
- Penalizes confident wrong predictions heavily (log(0) → -inf).

Numerical stability:
- Clip predictions to [eps, 1-eps] to avoid log(0).
- For logits, use the log-sum-exp trick (see cross_entropy_logits.py).
"""

import numpy as np


def cross_entropy(y_pred, y_true, eps=1e-15):
    """Compute cross-entropy loss from probability distributions.

    Args:
        y_pred: predicted probabilities, shape (..., C)
        y_true: true probabilities (one-hot), shape (..., C)
        eps:    small constant to prevent log(0).

    Returns:
        Scalar loss (mean over all samples).
    """
    # Clip to avoid log(0) which is -inf.
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # CE = -sum(y_true * log(y_pred))
    return -np.sum(y_true * np.log(y_pred), axis=-1)


def cross_entropy_sparse(probs, labels, eps=1e-15):
    """Compute cross-entropy loss from probabilities with sparse labels.

    Args:
        probs:  predicted probabilities, shape (B, C)
        labels: true class indices, shape (B,)
        eps:    small constant to prevent log(0).

    Returns:
        Scalar mean loss.
    """
    m = len(labels)
    probs = np.clip(probs, eps, 1 - eps)

    # Extract probability of the correct class for each sample.
    correct_probs = probs[np.arange(m), labels]

    return -np.mean(np.log(correct_probs))


# --- Demo ---
if __name__ == "__main__":
    # 3 classes, 6 samples
    logits = np.array([
        [2.0, 0.5, 0.1],
        [2.2, 0.4, 0.2],
        [0.2, 2.0, 0.5],
        [0.3, 2.2, 0.4],
        [0.1, 0.5, 2.0],
        [0.2, 0.4, 2.2]
    ])
    y_true = np.array([0, 0, 1, 1, 2, 2])

    # Convert logits to probabilities via softmax.
    from softmax import softmax
    probs = softmax(logits)

    print(f"Cross-entropy (sparse): {cross_entropy_sparse(probs, y_true):.4f}")

    # Confidently wrong case: model predicts class 2, true is class 0.
    logits_wrong = np.array([
        [0.1, 0.5, 2.0],  # model thinks class 2
        [2.0, 0.5, 0.1],  # model thinks class 0
    ])
    y_wrong = np.array([0, 1])

    probs_wrong = softmax(logits_wrong)
    print(f"Confidently wrong loss: {cross_entropy_sparse(probs_wrong, y_wrong):.4f}")
