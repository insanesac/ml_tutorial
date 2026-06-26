"""Minimal NumPy Transformer (GPT-style) implementation.

This file is intentionally educational:
- Shows the core GPT forward pass components in small functions.
- Keeps tensor shapes explicit so debugging is easier.
- Explains *why* each operation exists (not only what it does).

Architecture overview:
    tokens → embedding → positional encoding → [transformer blocks] × L → vocab projection → logits

    Each transformer block:
    1. Multi-head attention + residual + layer norm
    2. Feed-forward network + residual + layer norm
"""

import numpy as np


# --------------------------------------------------
# Softmax
# --------------------------------------------------

def softmax(x):
    """Compute softmax over the last axis.

    Why axis=-1?
    - In attention, scores are (..., query_position, key_position).
    - We want each query position to produce a probability distribution over keys.
    - So we normalize across the key dimension (last axis).

    Why subtract max first?
    - Numerical stability: exp(large number) can overflow.
    - softmax(x) == softmax(x + c), so shifting by -max is safe.
    """
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# --------------------------------------------------
# LayerNorm
# --------------------------------------------------

def layer_norm(x, eps=1e-5):
    """Apply Layer Normalization over the embedding dimension.

    For input shape (B, N, D):
    - mean/var are computed over D (axis=-1).
    - Each token vector is normalized independently.

    Why LayerNorm in transformers?
    - Stabilizes training by controlling activation scale.
    - Works at batch size 1 (important for autoregressive inference).
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# --------------------------------------------------
# Causal Mask
# --------------------------------------------------

def causal_mask(N):
    """Create causal (look-ahead) mask of shape (N, N).

    mask[i, j] = 0 if j <= i (allowed), -inf if j > i (blocked).
    Adding this to attention scores before softmax makes future tokens
    get probability ~0 after softmax.
    """
    mask = np.triu(np.ones((N, N)), k=1)
    mask = np.where(mask == 1, -np.inf, 0)
    return mask


# --------------------------------------------------
# Scaled Dot Product Attention
# --------------------------------------------------

def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Args:
        Q, K, V: shape (B, H, N, d_k)
        mask:    shape (N, N) or (1, 1, N, N), additive.

    Returns:
        Output of shape (B, H, N, d_k).

    Why divide by sqrt(d_k)?
    - Dot products grow with dimension, which can saturate softmax.
    - Scaling keeps gradients better behaved.
    """
    d_k = Q.shape[-1]

    # (B, H, N, d_k) @ (B, H, d_k, N) → (B, H, N, N)
    scores = Q @ K.transpose(0, 1, 3, 2)
    scores /= np.sqrt(d_k)

    if mask is not None:
        scores += mask

    # Attention weights: for each query, a distribution over keys.
    A = softmax(scores)

    # Weighted sum of value vectors.
    return A @ V


# --------------------------------------------------
# Multi Head Attention
# --------------------------------------------------

def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    """Multi-head self-attention block.

    Steps:
    1. Project X to Q, K, V in model dimension D.
    2. Split into num_heads heads (each head_dim = D / num_heads).
    3. Run scaled dot-product attention per head.
    4. Concatenate heads back to D.
    5. Final output projection Wo.

    Why multiple heads?
    - Different heads can learn different relationships (syntax, position,
      long-range dependency, etc.) in parallel.
    """
    B, N, D = X.shape
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
    out = attention(Q, K, V, mask)

    # Concatenate heads: (B, H, N, head_dim) → (B, N, D)
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)

    # Final output projection.
    return out @ Wo


# --------------------------------------------------
# Feed Forward Network
# --------------------------------------------------

def feed_forward(X, W1, b1, W2, b2):
    """Position-wise feed-forward network.

    Applied independently at each token position.
    Pattern: Linear → ReLU → Linear.
    Typically expands to 4x hidden dimension then projects back.
    """
    # First affine layer + ReLU nonlinearity.
    hidden = np.maximum(0, X @ W1 + b1)

    # Project back to model dimension.
    return hidden @ W2 + b2


# --------------------------------------------------
# Transformer Block
# --------------------------------------------------

def transformer_block(X, Wq, Wk, Wv, Wo, W1, b1, W2, b2, num_heads, mask):
    """Single transformer block (post-norm style with residuals).

    Structure:
    - Multi-head attention + residual + layer norm
    - Feed-forward + residual + layer norm

    Residual connections help gradients flow through deep stacks.
    """
    # Sub-layer 1: attention + residual + norm.
    mha_out = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask)
    X = layer_norm(X + mha_out)

    # Sub-layer 2: FFN + residual + norm.
    ffn_out = feed_forward(X, W1, b1, W2, b2)
    X = layer_norm(X + ffn_out)

    return X


# --------------------------------------------------
# GPT Forward Pass
# --------------------------------------------------

def gpt_forward(token_ids, token_embedding, position_embedding,
                transformer_blocks, W_vocab):
    """Run GPT-style forward pass to produce next-token logits.

    Args:
        token_ids:         (B, N) — input token indices.
        token_embedding:   (V, D) — embedding lookup table.
        position_embedding:(N_max, D) — learned positional embeddings.
        transformer_blocks: iterable of callables block(X, mask).
        W_vocab:           (D, V) — final projection to vocabulary.

    Returns:
        logits: (B, N, V) — next-token logits at each position.

    Why logits (not probabilities)?
    - Training loss (cross-entropy) is computed from logits directly
      for numerical stability (log-sum-exp trick).
    """
    B, N = token_ids.shape

    # Embedding lookup: (B, N) → (B, N, D)
    X = token_embedding[token_ids]

    # Positional encoding: inject token order information.
    # Attention alone is permutation-invariant; positions break this symmetry.
    # position_embedding[:N] has shape (N, D), broadcasts across batch.
    X = X + position_embedding[:N]

    # Causal mask: prevent attending to future tokens.
    mask = causal_mask(N)

    # Transformer layers: each block updates contextual representations.
    for block in transformer_blocks:
        X = block(X, mask)

    # Vocabulary projection: (B, N, D) @ (D, V) → (B, N, V)
    logits = X @ W_vocab

    return logits


# --------------------------------------------------
# Architecture diagram
# --------------------------------------------------
"""
token_ids (B, N)
    ↓
embedding lookup → (B, N, D)
    ↓
+ positional encoding → (B, N, D)
    ↓
[masked multi-head attention → add & norm]  ┐
    ↓                                        │ repeat L times
[feed-forward network → add & norm]         ┘
    ↓
final hidden states → (B, N, D)
    ↓
linear vocab projection → (B, N, V)
    ↓
logits
"""
