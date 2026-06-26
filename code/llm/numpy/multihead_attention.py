"""Multi-Head Attention (MHA) in NumPy.

MHA splits the model dimension D into H heads, each of size head_dim = D / H.
Each head independently attends to different aspects of the input.

Steps:
1. Project X → Q, K, V (each shape (B, N, D))
2. Reshape to (B, H, N, head_dim) — split into heads
3. Run scaled dot-product attention per head
4. Concatenate heads back to (B, N, D)
5. Final output projection Wo

Why multiple heads?
- Different heads can learn different relationships (syntax, position,
  long-range dependency, etc.) in parallel.
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention for 4D tensors.

    Args:
        Q, K, V: shape (B, H, N, head_dim)
        mask: shape (N, N) or (1, 1, N, N), additive.

    Returns:
        Output of shape (B, H, N, head_dim).
    """
    d_k = Q.shape[-1]

    # (B, H, N, d_k) @ (B, H, d_k, N) → (B, H, N, N)
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)

    # (B, H, N, N) @ (B, H, N, head_dim) → (B, H, N, head_dim)
    return weights @ V


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """Multi-head self-attention.

    Args:
        X:   input, shape (B, N, D)
        Wq:  query projection, shape (D, D)
        Wk:  key projection,   shape (D, D)
        Wv:  value projection, shape (D, D)
        Wo:  output projection, shape (D, D)
        num_heads: number of attention heads H
        mask: optional additive mask, shape (N, N)

    Returns:
        Output of shape (B, N, D)
    """
    B, N, D = X.shape
    assert D % num_heads == 0, f"D={D} must be divisible by num_heads={num_heads}"
    head_dim = D // num_heads

    # Linear projections: (B, N, D) @ (D, D) → (B, N, D)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Reshape (B, N, D) → (B, N, H, head_dim) → transpose → (B, H, N, head_dim)
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention per head: (B, H, N, head_dim)
    out = scaled_dot_product_attention(Q, K, V, mask)

    # Concatenate heads: (B, H, N, head_dim) → (B, N, H * head_dim) = (B, N, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

    # Final output projection: (B, N, D) @ (D, D) → (B, N, D)
    return out @ Wo
