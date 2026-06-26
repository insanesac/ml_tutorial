"""Scaled dot-product attention in NumPy.

Attention mechanism: given queries Q, keys K, and values V,
compute a weighted sum of V where weights come from Q-K similarity.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Why divide by sqrt(d_k)?
- Dot products grow with dimension d_k, which can saturate softmax.
- Scaling keeps the score magnitudes controlled, leading to better gradients.

Causal masking:
- In autoregressive models, token t must not attend to future tokens > t.
- We set future positions to -inf before softmax → probability ~0.
"""

import numpy as np


def causal_mask(N):
    """Create a causal (look-ahead) mask of shape (N, N).

    mask[i, j] = 0 if j <= i (allowed), -inf if j > i (blocked).

    Args:
        N: sequence length.

    Returns:
        Array of shape (N, N) — additive mask for attention scores.
    """
    # Upper triangle above diagonal = future positions.
    mask = np.triu(np.ones((N, N)), k=1)

    # Convert binary mask into additive mask: 0 for allowed, -inf for blocked.
    mask = np.where(mask == 1, -1e9, 0)

    return mask


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention.

    Args:
        Q: queries, shape (N, d_k) or (B, H, N, d_k)
        K: keys,    shape (N, d_k) or (B, H, N, d_k)
        V: values,  shape (N, d_v) or (B, H, N, d_v)
        mask: optional additive mask, shape (N, N) or broadcastable.

    Returns:
        Output of shape (N, d_v) or (B, H, N, d_v) — weighted sum of V.
    """
    d_k = Q.shape[-1]

    # Similarity: dot product of each query with each key.
    # For 2D: (N, d_k) @ (d_k, N) → (N, N)
    # For 4D: (B, H, N, d_k) @ (B, H, d_k, N) → (B, H, N, N)
    if Q.ndim == 2:
        scores = Q @ K.T
    else:
        scores = Q @ K.transpose(0, 1, 3, 2)

    # Scale by sqrt(d_k) to prevent softmax saturation.
    scores = scores / np.sqrt(d_k)

    # Apply causal mask: future positions get -inf → softmax gives ~0.
    if mask is not None:
        scores = scores + mask

    # Attention weights: probability distribution over keys for each query.
    weights = softmax(scores)

    # Weighted sum of value vectors.
    return weights @ V


def softmax(x):
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# --- Demo ---
if __name__ == "__main__":
    N, d_k, d_v = 4, 8, 8

    Q = np.random.randn(N, d_k)
    K = np.random.randn(N, d_k)
    V = np.random.randn(N, d_v)

    # Without mask (full attention)
    out = scaled_dot_product_attention(Q, K, V)
    print(f"Attention output shape: {out.shape}")

    # With causal mask (autoregressive)
    mask = causal_mask(N)
    out_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    print(f"Masked attention output shape: {out_masked.shape}")
