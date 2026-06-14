import numpy as np


# --------------------------------------------------
# Softmax
# --------------------------------------------------

def softmax(x):

    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


# --------------------------------------------------
# LayerNorm
# --------------------------------------------------

def layer_norm(x, eps=1e-5):

    mean = np.mean(
        x,
        axis=-1,
        keepdims=True
    )

    var = np.var(
        x,
        axis=-1,
        keepdims=True
    )

    return (
        x - mean
    ) / np.sqrt(
        var + eps
    )


# --------------------------------------------------
# Causal Mask
# --------------------------------------------------

def causal_mask(N):

    mask = np.triu(
        np.ones((N, N)),
        k=1
    )

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

    d_k = Q.shape[-1]

    scores = (
        Q @ K.transpose(
            0, 1, 3, 2
        )
    )

    scores /= np.sqrt(d_k)

    if mask is not None:
        scores += mask

    A = softmax(scores)

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

    B, N, D = X.shape

    head_dim = D // num_heads

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

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

    hidden = np.maximum(
        0,
        X @ W1 + b1
    )

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

    B, N = token_ids.shape

    # Embedding lookup

    X = token_embedding[
        token_ids
    ]

    # Positional Encoding

    X = (
        X
        +
        position_embedding[:N]
    )

    # Causal Mask

    mask = causal_mask(N)

    # Transformer Layers

    for block in transformer_blocks:

        X = block(
            X,
            mask
        )

    # Vocabulary Projection

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