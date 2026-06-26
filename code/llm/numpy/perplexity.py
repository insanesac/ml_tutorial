"""Perplexity in NumPy.

Perplexity (PPL) is the exponentiation of cross-entropy:
    PPL = exp(CE)

Interpretation:
- PPL measures how "confused" the model is.
- If PPL = k, the model is as uncertain as if choosing uniformly among k options.
- Lower PPL = better model.
- PPL of 1.0 = perfect prediction (CE = 0).

Example:
- CE = 2.0 → PPL = e^2 ≈ 7.39 (model is as confused as picking from ~7 options)
- CE = 0.5 → PPL = e^0.5 ≈ 1.65 (model is fairly confident)
"""

import numpy as np


def perplexity(cross_entropy_loss):
    """Compute perplexity from cross-entropy loss.

    Args:
        cross_entropy_loss: scalar or array of CE losses.

    Returns:
        Perplexity = exp(CE).
    """
    return np.exp(cross_entropy_loss)


# --- Demo ---
if __name__ == "__main__":
    ce_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    for ce in ce_values:
        print(f"CE = {ce:.1f} → PPL = {perplexity(ce):.2f}")
