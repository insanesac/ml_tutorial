"""Rotary Positional Embeddings (RoPE) in NumPy.

RoPE rotates Q and K vectors in 2D subspaces based on their position:
    For each pair (x_2i, x_2i+1), apply a rotation by angle θ_i * position:

    [x'_2i  ]   [cos(θ_i)  -sin(θ_i)] [x_2i  ]
    [x'_2i+1] = [sin(θ_i)   cos(θ_i)] [x_2i+1]

where θ_i = 10000^(-2i/d) — different frequency per dimension pair.

Why RoPE?
- Encodes relative position: the angle between Q_t and K_s depends on (t - s).
- No learned parameters needed (just sin/cos of positions).
- Extrapolates to longer sequences better than learned positional embeddings.
- Used in LLaMA, Mistral, Gemma, PaLM, etc.

Key properties:
- head_dim must be even (we process pairs of dimensions).
- cos and sin are precomputed once (they don't depend on input data).
- Only applied to Q and K (not V) — position affects attention scores, not values.
"""

import numpy as np


def build_rope(seq_len, head_dim):
    """Precompute cos and sin tables for RoPE.

    Args:
        seq_len:  maximum sequence length.
        head_dim: dimension of each attention head (must be even).

    Returns:
        cos: shape (1, 1, seq_len, head_dim/2)
        sin: shape (1, 1, seq_len, head_dim/2)
    """
    # Inverse frequency: θ_i = 10000^(-2i/d) for i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (10000 ** (np.arange(0, head_dim, 2).astype(np.float64) / head_dim))

    # Position indices: 0, 1, 2, ..., seq_len-1
    positions = np.arange(seq_len).astype(np.float64)

    # Outer product: angles[pos, i] = position * inv_freq[i]
    angles = np.outer(positions, inv_freq)

    cos = np.cos(angles)
    sin = np.sin(angles)

    # Add batch and head dims for broadcasting: (1, 1, seq_len, head_dim/2)
    return cos[np.newaxis, np.newaxis, :, :], sin[np.newaxis, np.newaxis, :, :]


def apply_rope(x, cos, sin):
    """Apply rotary positional embedding to a tensor.

    Args:
        x:   input tensor, shape (B, H, T, head_dim) — Q or K.
        cos: precomputed cos table, shape (1, 1, T, head_dim/2).
        sin: precomputed sin table, shape (1, 1, T, head_dim/2).

    Returns:
        Rotated tensor of same shape as x.
    """
    # Split into even and odd indexed elements (each of size head_dim/2).
    even = x[..., ::2]   # (B, H, T, head_dim/2)
    odd = x[..., 1::2]   # (B, H, T, head_dim/2)

    # Apply 2D rotation to each pair.
    # [cos -sin] [even]   [even*cos - odd*sin]
    # [sin  cos] [odd]  = [even*sin + odd*cos]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    # Interleave back into original layout.
    output = np.empty_like(x)
    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd

    return output


# --- Demo ---
if __name__ == "__main__":
    B, H, T, head_dim = 1, 4, 8, 64

    # Precompute cos/sin (done once, reused for all forward passes).
    cos, sin = build_rope(T, head_dim)
    print(f"cos shape: {cos.shape}")  # (1, 1, 8, 32)

    # Apply to Q and K (not V!).
    Q = np.random.randn(B, H, T, head_dim)
    K = np.random.randn(B, H, T, head_dim)

    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    print(f"Q_rot shape: {Q_rot.shape}")  # (1, 4, 8, 64)
