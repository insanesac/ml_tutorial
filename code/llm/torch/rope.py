"""Rotary Positional Embeddings (RoPE) — functional implementation in PyTorch.

RoPE rotates Q and K vectors in 2D subspaces based on position:
    For each pair (x_2i, x_2i+1), apply rotation by angle θ_i * position:

    [x'_2i  ]   [cos(θ_i)  -sin(θ_i)] [x_2i  ]
    [x'_2i+1] = [sin(θ_i)   cos(θ_i)] [x_2i+1]

where θ_i = 10000^(-2i/d).

This is the functional (non-module) version. For the nn.Module version,
see rope_module.py.

Key points:
- head_dim must be even (we process pairs).
- cos/sin precomputed once (don't depend on input data).
- Only applied to Q and K (not V) — position affects attention scores only.
- [cos, -sin; sin, cos] is a standard 2D rotation matrix.
"""

import torch


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
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Position indices: 0, 1, 2, ..., seq_len-1
    positions = torch.arange(seq_len).float()

    # Outer product: angles[pos, i] = position * inv_freq[i]
    angles = torch.outer(positions, inv_freq)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Add batch and head dims for broadcasting: (1, 1, seq_len, head_dim/2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


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

    # Apply 2D rotation: [cos -sin; sin cos] @ [even; odd]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    # Interleave back into original layout.
    output = torch.empty_like(x)
    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd

    return output


def apply_rope_to_qk(seq_len, head_dim, q, k):
    """Convenience function: build RoPE tables and apply to both Q and K.

    Args:
        seq_len:  sequence length T.
        head_dim: dimension of each attention head.
        q:        queries, shape (B, H, T, head_dim).
        k:        keys,    shape (B, H, T, head_dim).

    Returns:
        (q_rotated, k_rotated) — same shapes as inputs.
    """
    cos, sin = build_rope(seq_len, head_dim)
    q_rotated = apply_rope(q, cos, sin)
    k_rotated = apply_rope(k, cos, sin)
    return q_rotated, k_rotated


# --- Demo ---
if __name__ == "__main__":
    B, H, T, head_dim = 1, 4, 8, 64

    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)

    q_rot, k_rot = apply_rope_to_qk(T, head_dim, q, k)
    print(f"q_rot shape: {q_rot.shape}")  # (1, 4, 8, 64)
    print(f"k_rot shape: {k_rot.shape}")  # (1, 4, 8, 64)
