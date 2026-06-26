"""Grouped-Query Attention (GQA) in PyTorch.

GQA shares KV heads across query heads to reduce KV cache memory.
- Q has num_heads heads, K and V have num_kv_heads heads.
- Each KV head serves group_size = num_heads / num_kv_heads query heads.
- repeat_interleave expands KV heads to match Q heads.

KV cache memory savings:
- MHA: H KV heads → H * head_dim * T per layer
- GQA: num_kv_heads KV heads → num_kv_heads * head_dim * T per layer
- Example: H=32, num_kv_heads=8 → 4x reduction

Used in: LLaMA-2 70B, LLaMA-3, Mistral, Gemma.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    """Grouped-Query Attention module.

    Args:
        d_model:      model embedding dimension D.
        num_heads:    number of query heads H.
        num_kv_heads: number of KV heads (must divide H evenly).
    """

    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Q projection: full D output (H heads).
        self.q_proj = nn.Linear(d_model, d_model)

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.head_dim = d_model // num_heads
        self.group_size = num_heads // num_kv_heads

        # K, V projections: only num_kv_heads * head_dim output (smaller than Q).
        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads)

        # Output projection: D → D.
        self.out_proj = nn.Linear(d_model, d_model)

    def causal_mask(self, T):
        """Create causal mask for sequence length T.

        Returns:
            Mask of shape (T, T) with -inf above diagonal, 0 elsewhere.
        """
        return torch.triu(
            torch.full((T, T), float("-inf"), device=self.q_proj.weight.device),
            diagonal=1
        )

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

        A = F.softmax(scores, dim=-1)

        # (B, H, N, N) @ (B, H, N, head_dim) → (B, H, N, head_dim)
        return torch.matmul(A, V)

    def forward(self, x):
        """GQA forward pass.

        Args:
            x: input, shape (B, T, D).

        Returns:
            Output of shape (B, T, D).
        """
        B, T, D = x.shape

        # Create causal mask for this sequence length.
        mask = self.causal_mask(T)

        # Linear projections.
        Q = self.q_proj(x)  # (B, T, D)
        K = self.k_proj(x)  # (B, T, num_kv_heads * head_dim)
        V = self.v_proj(x)  # (B, T, num_kv_heads * head_dim)

        # Reshape to heads.
        # Q: (B, T, D) → (B, T, H, head_dim) → (B, H, T, head_dim)
        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: (B, T, num_kv_heads * head_dim) → (B, T, num_kv_heads, head_dim) → (B, num_kv_heads, T, head_dim)
        K = K.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads via repeat_interleave.
        # Each KV head is repeated group_size times consecutively.
        # (B, num_kv_heads, T, head_dim) → (B, H, T, head_dim)
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        # Attention.
        output = self.attention(Q, K, V, mask)

        # Concatenate heads: (B, H, T, head_dim) → (B, T, D)
        output = output.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(output)
