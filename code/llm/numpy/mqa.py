"""Multi-Query Attention (MQA) in NumPy.

MQA shares a single K and V head across all query heads.
- Q has H heads, but K and V have only 1 head each.
- K and V are repeated H times to match Q's head dimension.

Why MQA?
- KV cache memory is reduced by a factor of H (only 1 KV head instead of H).
- Slight quality drop vs MHA, but significant memory and speed improvement.

Shapes:
  Q: (B, H, N, head_dim)  — H query heads
  K: (B, 1, N, head_dim)  — 1 shared KV head
  V: (B, 1, N, head_dim)  — 1 shared KV head
  → K, V expanded to (B, H, N, head_dim) via np.repeat
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def mqa(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """Multi-Query Attention.

    Args:
        X:   input, shape (B, N, D)
        Wq:  query projection, shape (D, D)       — full D output
        Wk:  key projection,   shape (D, head_dim) — only 1 head
        Wv:  value projection, shape (D, head_dim) — only 1 head
        Wo:  output projection, shape (D, D)
        num_heads: number of query heads H
        mask: optional additive mask

    Returns:
        Output of shape (B, N, D)
    """
    B, N, D = X.shape
    assert D % num_heads == 0
    head_dim = D // num_heads

    # Projections: Q is full (B, N, D), K and V are (B, N, head_dim)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Q: (B, N, D) → (B, N, H, head_dim) → (B, H, N, head_dim)
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # K, V: (B, N, head_dim) → (B, N, 1, head_dim) → (B, 1, N, head_dim)
    K = K.reshape(B, N, 1, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, 1, head_dim).transpose(0, 2, 1, 3)

    # Expand single KV head to match H query heads: (B, 1, N, head_dim) → (B, H, N, head_dim)
    K = np.repeat(K, num_heads, axis=1)
    V = np.repeat(V, num_heads, axis=1)

    # Scaled dot-product attention.
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)
    out = weights @ V

    # Concatenate heads: (B, H, N, head_dim) → (B, N, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

    return out @ Wo
