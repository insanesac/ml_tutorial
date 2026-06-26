"""Transformer Block in PyTorch.

A single transformer block (post-norm style) containing:
1. Multi-head self-attention + residual + LayerNorm
2. Feed-forward network (Linear → ReLU → Linear) + residual + LayerNorm

This is the building block of GPT-style models. Stack L of these
to form a full transformer.

Shapes (assuming input X of shape (B, N, D)):
- Q, K, V projections: (B, N, D) → (B, N, D)
- Reshape to heads: (B, N, D) → (B, H, N, head_dim)
- Attention scores: (B, H, N, N)
- Attention output: (B, H, N, head_dim) → (B, N, D)
- FFN hidden: (B, N, 4*D)
- Output: (B, N, D)
"""

import math
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """A single transformer encoder/decoder block.

    Args:
        d_model:    model embedding dimension D.
        num_heads:  number of attention heads H.
        vocab_size: vocabulary size (unused here, kept for compatibility).
        max_seq_len: maximum sequence length (unused here, kept for compatibility).
    """

    def __init__(self, d_model, num_heads, vocab_size=None, max_seq_len=None):
        super().__init__()

        # Q, K, V projections: D → D
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection after concatenating heads: D → D
        self.out_proj = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"d_model={d_model} must be divisible by num_heads={num_heads}"
        self.head_dim = d_model // num_heads

        # Position-wise FFN: D → 4*D → D
        self.ffn1 = nn.Linear(d_model, 4 * d_model)
        self.ffn2 = nn.Linear(4 * d_model, d_model)

        # LayerNorm for each sub-layer.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def softmax(self, x):
        """Numerically stable softmax over the last axis."""
        x = x - torch.max(x, dim=-1, keepdim=True).values
        exp = torch.exp(x)
        return exp / torch.sum(exp, dim=-1, keepdim=True)

    def attention(self, Q, K, V, mask=None):
        """Scaled dot-product attention.

        Args:
            Q, K, V: shape (B, H, N, head_dim)
            mask:    additive mask, shape (N, N) or (1, 1, N, N).

        Returns:
            Output of shape (B, H, N, head_dim).
        """
        d_k = Q.shape[-1]

        # (B, H, N, head_dim) @ (B, H, head_dim, N) → (B, H, N, N)
        scores = torch.matmul(Q, K.transpose(2, 3))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        A = self.softmax(scores)

        # (B, H, N, N) @ (B, H, N, head_dim) → (B, H, N, head_dim)
        return torch.matmul(A, V)

    def mha(self, X, mask=None):
        """Multi-head attention forward pass.

        Args:
            X:    input, shape (B, N, D).
            mask: additive mask, shape (N, N).

        Returns:
            Output of shape (B, N, D).
        """
        B, N, D = X.shape

        # Linear projections: (B, N, D) → (B, N, D)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # Reshape to heads: (B, N, D) → (B, N, H, head_dim) → (B, H, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: (B, H, N, head_dim)
        output = self.attention(Q, K, V, mask)

        # Concatenate heads: (B, H, N, head_dim) → (B, N, D)
        output = output.transpose(1, 2).reshape(B, N, D)

        return self.out_proj(output)

    def ffn_layer(self, X):
        """Position-wise feed-forward network: Linear → ReLU → Linear."""
        hidden = torch.relu(self.ffn1(X))
        return self.ffn2(hidden)

    def forward(self, X, mask=None):
        """Full transformer block forward pass.

        Sub-layer 1: MHA + residual + LayerNorm
        Sub-layer 2: FFN + residual + LayerNorm
        """
        mha_out = self.mha(X, mask)
        X = self.norm1(mha_out + X)

        ffn_out = self.ffn_layer(X)
        X = self.norm2(X + ffn_out)

        return X
