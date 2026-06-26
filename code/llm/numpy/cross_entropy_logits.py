"""Cross-entropy loss computed directly from logits in NumPy.

Computing CE from logits (instead of probabilities) is more numerically stable.
Uses the log-sum-exp (LSE) trick:

    CE = -log(softmax(logits)[correct_class])
       = -logits[correct] + log(sum(exp(logits)))
       = -logits[correct] + LSE(logits)

With max-shifting for stability:
    LSE(logits) = max(logits) + log(sum(exp(logits - max(logits)))

This avoids computing softmax explicitly, preventing underflow/overflow.
"""

import numpy as np


def cross_entropy_from_logits(logits, labels):
    """Compute mean cross-entropy loss from logits using log-sum-exp trick.

    Args:
        logits: raw model outputs, shape (B, C)
        labels: true class indices, shape (B,)

    Returns:
        Scalar mean loss.
    """
    # Max-shift for numerical stability: subtract max from each row.
    max_logits = np.max(logits, axis=-1, keepdims=True)

    # Shifted logits (max becomes 0 → exp(0) = 1, no overflow).
    shifted = logits - max_logits

    # Log-sum-exp: log(sum(exp(shifted))) + max_logits
    # The + max_logits cancels out the shift, giving the true LSE.
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1)) + max_logits.squeeze(-1)

    # Logit of the correct class for each sample.
    z_correct = logits[np.arange(logits.shape[0]), labels]

    # CE = -z_correct + LSE(logits)
    loss = -z_correct + logsumexp

    return np.mean(loss)


# --- Demo ---
if __name__ == "__main__":
    logits = np.array([
        [2.0, 0.5, 0.1],
        [0.1, 0.5, 2.0],  # confidently wrong for class 0
    ])
    labels = np.array([0, 0])

    print(f"CE from logits: {cross_entropy_from_logits(logits, labels):.4f}")
