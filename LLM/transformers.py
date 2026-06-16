"""Minimal NumPy Transformer implementation with heavy explanations.

This file is intentionally educational:
- Shows the core GPT forward pass components in small functions.
- Keeps tensor shapes explicit so debugging is easier.
- Explains *why* each operation exists (not only what it does).
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

    # Shift values so the largest element becomes 0; prevents exp overflow.
    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    # Convert shifted logits to positive (unnormalized) scores.
    exp_x = np.exp(x)

    # Normalize to probabilities that sum to 1 along the last axis.
    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


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

    # Token-wise mean over features.
    mean = np.mean(
        x,
        axis=-1,
        keepdims=True
    )

    # Token-wise variance over features.
    var = np.var(
        x,
        axis=-1,
        keepdims=True
    )

    # Normalize to zero mean and unit variance (approximately).
    return (
        x - mean
    ) / np.sqrt(
        var + eps
    )


# --------------------------------------------------
# Causal Mask
# --------------------------------------------------

def causal_mask(N):
    """Create causal (look-ahead) mask of shape (N, N).

    Meaning of mask values:
    - 0: attention allowed
    - -inf: attention blocked

    Adding this to attention scores before softmax makes future tokens
    get probability ~0 after softmax.
    """

    # Upper triangle above diagonal = future positions.
    mask = np.triu(
        np.ones((N, N)),
        k=1
    )

    # Convert binary mask into additive mask for scores.
    mask = np.where(
        mask == 1,
        -np.inf,
        0
    )

    return mask


# --------------------------------------------------
# Scaled Dot Product Attention
# --------------------------------------------------

def attention(
    Q,
    K,
    V,
    mask=None
):
    """Scaled dot-product attention.

    Expected shapes (multi-head case):
    - Q, K, V: (B, H, N, d_k)
    - scores:   (B, H, N, N)
    - output:   (B, H, N, d_k)

    Why divide by sqrt(d_k)?
    - Dot products grow with dimension, which can saturate softmax.
    - Scaling keeps gradients better behaved.
    """

    # Head dimension used for scale factor.
    d_k = Q.shape[-1]

    # Similarity of each query token to each key token.
    scores = (
        Q @ K.transpose(
            0, 1, 3, 2
        )
    )

    # Stabilize softmax by keeping score magnitudes controlled.
    scores /= np.sqrt(d_k)

    if mask is not None:
        # Broadcasting works with mask shape (N, N) or (1, 1, N, N).
        # Future-token entries become -inf so softmax turns them into ~0.
        scores += mask

    # Attention weights: for each query token, a distribution over keys.
    A = softmax(scores)

    # Weighted sum of value vectors.
    return A @ V


# --------------------------------------------------
# Multi Head Attention
# --------------------------------------------------

def multi_head_attention(
    X,
    Wq,
    Wk,
    Wv,
    Wo,
    num_heads,
    mask=None
):
    """Multi-head self-attention block.

    Steps:
    1) Project X to Q, K, V in model dimension D.
    2) Split into num_heads heads (each head_dim = D / num_heads).
    3) Run scaled dot-product attention per head.
    4) Concatenate heads back to D.
    5) Final output projection Wo.

    Why multiple heads?
    - Different heads can learn different relationships (syntax, position,
      long-range dependency, etc.) in parallel.
    """

    B, N, D = X.shape

    # Each head gets a subspace of the model dimension.
    head_dim = D // num_heads

    # Linear projections from model space -> query/key/value spaces.
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # Reshape from (B, N, D) -> (B, H, N, head_dim).
    # transpose puts head dimension before sequence for batched attention math.
    Q = Q.reshape(
        B,
        N,
        num_heads,
        head_dim
    ).transpose(
        0, 2, 1, 3
    )

    K = K.reshape(
        B,
        N,
        num_heads,
        head_dim
    ).transpose(
        0, 2, 1, 3
    )

    V = V.reshape(
        B,
        N,
        num_heads,
        head_dim
    ).transpose(
        0, 2, 1, 3
    )

    out = attention(
        Q,
        K,
        V,
        mask
    )

    # Reassemble heads: (B, H, N, head_dim) -> (B, N, D).
    out = out.transpose(
        0,
        2,
        1,
        3
    )

    out = out.reshape(
        B,
        N,
        D
    )

    # Final linear mix of head outputs.
    return out @ Wo


# --------------------------------------------------
# Feed Forward Network
# --------------------------------------------------

def feed_forward(
    X,
    W1,
    b1,
    W2,
    b2
):
    """Position-wise feed-forward network.

    Applied independently at each token position.
    Typical transformer pattern: Linear -> ReLU/GELU -> Linear.
    """

    # First affine layer + nonlinearity introduces capacity beyond attention.
    hidden = np.maximum(
        0,
        X @ W1 + b1
    )

    # Project back to model dimension.
    return (
        hidden @ W2 + b2
    )


# --------------------------------------------------
# Transformer Block
# --------------------------------------------------

def transformer_block(
    X,
    Wq,
    Wk,
    Wv,
    Wo,
    W1,
    b1,
    W2,
    b2,
    num_heads,
    mask
):
    """Single transformer block (pre-norm style here with residuals).

    Structure:
    - Multi-head attention + residual + layer norm
    - Feed-forward      + residual + layer norm

    Residual connections help gradients flow through deep stacks.
    """

    mha_out = multi_head_attention(
        X,
        Wq,
        Wk,
        Wv,
        Wo,
        num_heads,
        mask
    )

    X = layer_norm(
        X + mha_out
    )

    # Token-wise MLP transformation after attention mixing.
    ffn_out = feed_forward(
        X,
        W1,
        b1,
        W2,
        b2
    )

    X = layer_norm(
        X + ffn_out
    )

    return X


# --------------------------------------------------
# GPT Forward
# --------------------------------------------------

def gpt_forward(
    token_ids,
    token_embedding,
    position_embedding,
    transformer_blocks,
    W_vocab
):
    """Run GPT-style forward pass to produce next-token logits.

    Inputs:
    - token_ids: (B, N)
    - token_embedding: (V, D)
    - position_embedding: (N_max, D)
    - transformer_blocks: iterable of callables block(X, mask)
    - W_vocab: (D, V)

    Output:
    - logits: (B, N, V)

    Why logits (not probabilities)?
    - Training loss (cross-entropy) is usually computed from logits directly
      for numerical stability.
    """

    B, N = token_ids.shape

    # Embedding lookup
    # Converts token indices into dense vectors.
    # token_embedding[token_ids] uses advanced indexing to produce (B, N, D).

    X = token_embedding[
        token_ids
    ]

    # Positional Encoding
    # Inject token order information because attention alone is permutation-invariant.
    # position_embedding[:N] has shape (N, D) and broadcasts across batch B.

    X = (
        X
        +
        position_embedding[:N]
    )

    # Causal Mask
    # Prevent token t from attending to future tokens > t.

    mask = causal_mask(N)

    # Transformer Layers
    # Each block updates contextual representations while honoring causal mask.

    for block in transformer_blocks:

        X = block(
            X,
            mask
        )

    # Vocabulary Projection
    # Map hidden state at each position from D -> V to get token logits.

    logits = X @ W_vocab

    return logits


"""
token_ids
(B,N)

↓

embedding lookup

(B,N,D)

↓

position encoding

(B,N,D)

↓

multi-head attention

(B,N,D)

↓

FFN

(B,N,D)

↓

final hidden states

(B,N,D)

↓

vocab projection

(B,N,V)

"""

"""
tokens
↓
embedding lookup
↓
positional encoding
↓
masked multi-head attention
↓
add & norm
↓
feed-forward network
↓
add & norm
↓
repeat L times
↓
linear vocab projection
↓
logits
↓
softmax
↓
next token probabilities"""