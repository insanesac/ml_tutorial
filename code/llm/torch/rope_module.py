"""Rotary Positional Embeddings (RoPE) — nn.Module implementation in PyTorch.

This wraps the RoPE cos/sin tables as registered buffers in an nn.Module,
so they are automatically moved to the correct device with .to() / .cuda().

Registered buffers are not parameters (no gradients) but are part of
the module's state_dict (saved/loaded with checkpoints).
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """RoPE module with precomputed cos/sin buffers.

    Args:
        head_dim:       dimension of each attention head (must be even).
        max_seq_length: maximum sequence length for precomputed tables.
    """

    def __init__(self, head_dim, max_seq_length):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length

        # Precompute cos/sin tables and register as buffers.
        cos, sin = self._build_rope()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _build_rope(self):
        """Precompute cos and sin tables.

        Returns:
            cos: shape (1, 1, max_seq_length, head_dim/2)
            sin: shape (1, 1, max_seq_length, head_dim/2)
        """
        # Inverse frequency: θ_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

        # Position indices.
        positions = torch.arange(self.max_seq_length).float()

        # angles[pos, i] = position * inv_freq[i]
        angles = torch.outer(positions, inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Add batch and head dims for broadcasting.
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def _rotate(self, x):
        """Apply rotation to tensor x using precomputed cos/sin.

        Args:
            x: shape (B, H, T, head_dim).

        Returns:
            Rotated tensor of same shape.
        """
        T = x.shape[2]

        # Slice precomputed tables to current sequence length.
        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]

        # Split into even/odd pairs.
        even = x[..., ::2]
        odd = x[..., 1::2]

        # 2D rotation: [cos -sin; sin cos]
        pair1 = even * cos - odd * sin
        pair2 = even * sin + odd * cos

        # Interleave back.
        output = torch.empty_like(x)
        output[..., ::2] = pair1
        output[..., 1::2] = pair2

        return output

    def forward(self, q, k):
        """Apply RoPE to query and key tensors.

        Args:
            q: queries, shape (B, H, T, head_dim).
            k: keys,    shape (B, H, T, head_dim).

        Returns:
            (q_rotated, k_rotated) — same shapes as inputs.
        """
        q_rot = self._rotate(q)
        k_rot = self._rotate(k)
        return q_rot, k_rot


# --- Demo ---
if __name__ == "__main__":
    head_dim = 64
    max_seq_length = 2048

    rope = RoPE(head_dim, max_seq_length)

    B, H, T = 2, 8, 16
    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)

    q_rot, k_rot = rope(q, k)
    print(f"q_rot shape: {q_rot.shape}")  # (2, 8, 16, 64)
    print(f"k_rot shape: {k_rot.shape}")  # (2, 8, 16, 64)
