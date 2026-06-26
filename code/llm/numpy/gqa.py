"""Grouped-Query Attention (GQA) in NumPy.

GQA is a middle ground between MHA (H KV heads) and MQA (1 KV head).
- Q has H query heads, K and V have num_kv_heads (< H) KV heads.
- Each KV head is shared by group_size = H / num_kv_heads query heads.

Why GQA?
- KV cache memory reduced by factor of group_size vs MHA.
- Better quality than MQA (which is group_size = H).
- Used in LLaMA-2 70B, LLaMA-3, Mistral, etc.

Shapes:
  Q: (B, H, N, head_dim)         — H query heads
  K: (B, num_kv_heads, N, head_dim) — fewer KV heads
  V: (B, num_kv_heads, N, head_dim)
  → K, V repeated by group_size along head dim → (B, H, N, head_dim)
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax over the last axis."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def gqa(X, Wq, Wk, Wv, Wo, num_heads, num_kv_heads, mask=None):
    """Grouped-Query Attention.

    Args:
        X:   input, shape (B, N, D)
        Wq:  query projection, shape (D, D)
        Wk:  key projection,   shape (D, num_kv_heads * head_dim)
        Wv:  value projection, shape (D, num_kv_heads * head_dim)
        Wo:  output projection, shape (D, D)
        num_heads: number of query heads H
        num_kv_heads: number of KV heads (must divide H evenly)
        mask: optional additive mask

    Returns:
        Output of shape (B, N, D)
    """
    B, N, D = X.shape
    assert D % num_heads == 0
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

    head_dim = D // num_heads
    group_size = num_heads // num_kv_heads

    # Projections.
    Q = X @ Wq  # (B, N, D)
    K = X @ Wk  # (B, N, num_kv_heads * head_dim)
    V = X @ Wv  # (B, N, num_kv_heads * head_dim)

    # Q: (B, N, D) → (B, N, H, head_dim) → (B, H, N, head_dim)
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # K, V: (B, N, num_kv_heads * head_dim) → (B, N, num_kv_heads, head_dim) → (B, num_kv_heads, N, head_dim)
    K = K.reshape(B, N, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # Expand KV heads to match Q heads: each KV head serves group_size query heads.
    # (B, num_kv_heads, N, head_dim) → (B, H, N, head_dim)
    K = np.repeat(K, group_size, axis=1)
    V = np.repeat(V, group_size, axis=1)

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
