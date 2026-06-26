"""Multi-Query Attention (MQA) in PyTorch.

MQA is the extreme case of GQA where num_kv_heads = 1.
- Q has H heads, but K and V have only 1 head.
- The single KV head is expanded to match all H query heads.

KV cache memory savings vs MHA:
- MHA: H * head_dim * T per layer for K and V
- MQA: 1 * head_dim * T per layer for K and V
- Example: H=32 → 32x reduction in KV cache memory

Tradeoff: slight quality drop vs MHA, but significant speed/memory gain.
Used in: PaLM, Falcon, FlashAttention-optimized models.
"""

import math
import torch
import torch.nn as nn


class MQA(nn.Module):
    """Multi-Query Attention module.

    Args:
        d_model:   model embedding dimension D.
        num_heads: number of query heads H.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        # Q projection: full D output (H heads).
        self.q_proj = nn.Linear(d_model, d_model)

        # Output projection: D → D.
        self.out_proj = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        # K, V projections: only 1 head (head_dim output, not D).
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)

    def attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention.

        Args:
            Q, K, V: shape (B, H, N, head_dim)
            mask:    additive mask, shape (N, N).

        Returns:
            Output of shape (B, H, N, head_dim).
        """
        d_k = Q.shape[-1]

        # (B, H, N, head_dim) @ (B, H, head_dim, N) → (B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        A = torch.softmax(scores, dim=-1)

        return torch.matmul(A, V)

    def forward(self, x):
        """MQA forward pass.

        Args:
            x: input, shape (B, N, D).

        Returns:
            Output of shape (B, N, D).
        """
        B, N, D = x.shape

        # Linear projections.
        Q = self.q_proj(x)  # (B, N, D)
        K = self.k_proj(x)  # (B, N, head_dim) — only 1 head
        V = self.v_proj(x)  # (B, N, head_dim) — only 1 head

        # Q: (B, N, D) → (B, N, H, head_dim) → (B, H, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, N, head_dim) → (B, N, 1, head_dim) → (B, 1, N, head_dim)
        K = K.view(B, N, 1, self.head_dim).transpose(1, 2)
        V = V.view(B, N, 1, self.head_dim).transpose(1, 2)

        # Expand single KV head to match H query heads.
        # (B, 1, N, head_dim) → (B, H, N, head_dim)
        K = K.expand(-1, self.num_heads, -1, -1)
        V = V.expand(-1, self.num_heads, -1, -1)

        # Attention.
        output = self.attention(Q, K, V)

        # Concatenate heads: (B, H, N, head_dim) → (B, N, D)
        output = output.transpose(1, 2).reshape(B, N, D)

        return self.out_proj(output)
