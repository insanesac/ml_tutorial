"""GPT model in PyTorch.

A minimal GPT-style language model that stacks transformer blocks.

Architecture:
    tokens → token embedding + positional embedding
           → [TransformerBlock] × L
           → final LayerNorm
           → linear projection to vocabulary (logits)

Weight tying:
    The output projection (lm_head) shares weights with the token embedding.
    This reduces parameters and ensures input/output share the same semantic space.
"""

import importlib
import torch
import torch.nn as nn

# Import from numbered filename (can't use normal import with numeric prefix).
_transformer_block = importlib.import_module("00_transformer_block")
TransformerBlock = _transformer_block.TransformerBlock


class GPT(nn.Module):
    """GPT language model.

    Args:
        d_model:     model embedding dimension D.
        vocab_size:  number of tokens in vocabulary V.
        max_seq_len: maximum sequence length (for positional embedding).
        num_layers:  number of transformer blocks L.
        num_heads:   number of attention heads H.
    """

    def __init__(self, d_model, vocab_size, max_seq_len, num_layers, num_heads):
        super().__init__()

        # Token embedding: (V, D) — lookup table mapping token IDs to vectors.
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding: (max_seq_len, D) — encodes token position.
        # Without this, "I am cat" and "Cat am I" would be semantically identical
        # because attention is permutation-invariant.
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # Stack of L transformer blocks (parameters are shared via ModuleList).
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # Final LayerNorm before vocab projection.
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: D → V (produces logits for each token in vocab).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection.
        # This ensures the input and output share the same semantic space.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        """Forward pass producing next-token logits.

        Args:
            token_ids: shape (B, T) — input token indices.

        Returns:
            logits: shape (B, T, V) — next-token logits at each position.
        """
        B, T = token_ids.shape

        # Token embedding lookup: (B, T) → (B, T, D)
        X = self.token_embedding(token_ids)

        # Positional encoding: (T,) → (T, D), broadcasts across batch.
        positional_ids = torch.arange(T)
        pos = self.positional_embedding(positional_ids)
        X = X + pos

        # Pass through each transformer block.
        for block in self.blocks:
            X = block(X)

        # Final normalization.
        X = self.final_norm(X)

        # Vocabulary projection: (B, T, D) → (B, T, V)
        logits = self.lm_head(X)

        return logits
