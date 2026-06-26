"""Layer Normalization and RMSNorm in NumPy.

LayerNorm:
- Normalizes over the feature dimension (last axis) per token.
- mean and variance computed over D, independently for each token.
- Includes learnable scale (gamma) and shift (beta).
- Used in original Transformer (Vaswani et al., 2017).

RMSNorm:
- Simplification of LayerNorm: only normalizes by RMS (root mean square).
- No mean subtraction, no shift (beta).
- Only has learnable scale (gamma).
- Faster than LayerNorm (~10-50% fewer computations).
- Used in LLaMA, Mistral, Gemma, etc.

Why not BatchNorm?
- BatchNorm depends on batch size → breaks at batch=1 (autoregressive inference).
- LayerNorm/RMSNorm work at any batch size, including batch=1.
"""

import numpy as np


def layer_norm(x, gamma, beta, eps=1e-5):
    """Apply Layer Normalization over the last axis.

    Args:
        x:     input, shape (..., D)
        gamma: scale parameter, shape (D,)
        beta:  shift parameter, shape (D,)
        eps:   small constant for numerical stability.

    Returns:
        Normalized array of same shape as x.
    """
    # Per-token mean over features.
    mean = np.mean(x, axis=-1, keepdims=True)

    # Per-token variance over features.
    variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)

    # Standard deviation (add eps to avoid division by zero).
    std = np.sqrt(variance + eps)

    # Normalize to zero mean, unit variance.
    x_norm = (x - mean) / std

    # Apply learnable scale and shift.
    return gamma * x_norm + beta


def rms_norm(x, gamma, eps=1e-5):
    """Apply RMS Normalization over the last axis.

    RMSNorm = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)

    Key difference from LayerNorm:
    - No mean subtraction (assumes activations are centered).
    - No beta (shift) parameter.
    - Fewer computations → faster.

    Args:
        x:     input, shape (..., D)
        gamma: scale parameter, shape (D,)
        eps:   small constant for numerical stability.

    Returns:
        Normalized array of same shape as x.
    """
    # Root mean square over features.
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

    # Normalize by RMS and apply learnable scale.
    return gamma * (x / rms)


# --- Demo ---
if __name__ == "__main__":
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [10.0, 20.0, 30.0, 40.0]])

    gamma = np.ones(x.shape[-1])
    beta = np.zeros(x.shape[-1])

    print(f"LayerNorm:\n{layer_norm(x, gamma, beta)}")
    print(f"\nRMSNorm:\n{rms_norm(x, gamma)}")
