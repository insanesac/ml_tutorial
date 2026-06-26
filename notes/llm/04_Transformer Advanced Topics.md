# Transformer Advanced Topics

## Positional Encoding

### The Problem

Attention is **permutation invariant**.

It does not know token order. These are identical without position info:

```
"dog bites man"  ==  "man bites dog"   # to vanilla attention
```

### The Solution

Add positional embeddings to token embeddings:

```
X = token_embeddings + position_embeddings
```

### Shapes

```
token_embeddings:    (B, N, D)
position_embeddings: (N, D)   <- broadcasts over B
X:                   (B, N, D)
```

### Sinusoidal Encoding

- Even dimensions use `sin`
- Odd dimensions use `cos`
- Multiple frequencies allow the model to capture both local and global position

```python
def sinusoidal_encoding(N, D):
    PE = np.zeros((N, D))
    pos = np.arange(N)[:, None]
    div = np.exp(np.arange(0, D, 2) * (-np.log(10000) / D))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE
```

### Modern Alternative: RoPE

Rotary Positional Embeddings — used in LLaMA, Gemma, Mistral.

Encodes position by rotating Q and K vectors before attention. Better extrapolation to longer sequences than sinusoidal.

### Key Interview Points

- Attention alone is permutation invariant.
- Positional encoding injects order.
- Modern LLMs use RoPE, not sinusoidal.

---

## Self Attention vs Cross Attention

### Self Attention

Q, K, V all come from the **same** sequence:

```
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

Attention shape: `(B, N, N)`

Used in: GPT encoder blocks, BERT, every decoder self-attention layer.

### Cross Attention

Q comes from one sequence, K and V come from another:

```
Q = decoder_X @ Wq        # decoder queries
K = encoder_X @ Wk        # encoder keys
V = encoder_X @ Wv        # encoder values
```

### Cross Attention Shapes

| Tensor | Shape |
|---|---|
| Q | `(B, N_dec, d_k)` |
| K | `(B, N_enc, d_k)` |
| V | `(B, N_enc, d_v)` |
| Scores `Q @ Kᵀ` | `(B, N_dec, N_enc)` |
| Output | `(B, N_dec, d_v)` |

Each decoder token attends over **all** encoder tokens.

### Cross Attention Implementation

```python
def cross_attention(decoder_X, encoder_X, Wq, Wk, Wv, d_k):
    Q = decoder_X @ Wq
    K = encoder_X @ Wk
    V = encoder_X @ Wv

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    A = softmax(scores, axis=-1)
    return A @ V
```

### Complexity

`O(N_dec * N_enc * D)` — note N_dec and N_enc can differ.

### Translation Example

```
English -> Encoder -> K, V
French  -> Decoder -> Q
```

Decoder asks: "What English context is relevant to generate this French token?"

---

## Summary: Self vs Cross Attention

| | Self Attention | Cross Attention |
|---|---|---|
| Q source | Same tensor | Decoder |
| K, V source | Same tensor | Encoder |
| Score shape | `(B, N, N)` | `(B, N_dec, N_enc)` |
| Used in | GPT, BERT | Encoder-decoder (T5, BART) |

---

## Implementation Traps

### Positional Encoding

- Sequence length must match between `token_embeddings` and `position_embeddings`.
- Embedding dimension D must match.

### Cross Attention

- `N_dec` and `N_enc` can differ — score matrix is not square.
- Softmax always on `axis=-1` (over key dimension).
- `K` must be transposed before `Q @ Kᵀ`.

---

## ALiBi (Attention with Linear Biases)

### Core Idea

Instead of adding positional embeddings to the input, ALiBi adds a **linear distance penalty** directly to attention scores:

```
score(i, j) = (Q_i · K_j) / √d_k - m * |i - j|
```

Where `m` is a head-specific slope (geometrically decreasing across heads).

### Why ALiBi Is Interesting

- **No positional embeddings needed** — position is injected at the attention score level.
- **Strong extrapolation**: trained on short sequences, works on longer sequences without fine-tuning.
- **Simpler than RoPE**: no rotation, just a bias term.
- Used in: BLOOM, MPT.

### ALiBi vs RoPE

| | RoPE | ALiBi |
|---|---|---|
| Method | Rotate Q/K vectors | Add bias to attention scores |
| Extrapolation | Good (with NTK/YaRN scaling) | Excellent (native) |
| Used in | LLaMA, Gemma, Mistral | BLOOM, MPT |
| Complexity | O(N * D) rotation | O(N²) bias matrix (but cheap) |

---

## Pre-Norm vs Post-Norm

### Post-Norm (Original Transformer)

```
x = LayerNorm(x + SubLayer(x))
```

LayerNorm is applied **after** the residual connection.

- Used in: original Transformer paper, BERT.
- Problem: gradients can explode/vanish in deep models because the residual path goes through LayerNorm.
- Harder to train at scale — requires careful learning rate warmup.

### Pre-Norm (Modern LLMs)

```
x = x + SubLayer(LayerNorm(x))
```

LayerNorm is applied **before** the sublayer, and the residual connection is **clean** (no normalization on the skip path).

- Used in: GPT-2, GPT-3, LLaMA, all modern LLMs.
- Benefit: gradient flows freely through residual path — no normalization bottleneck.
- More stable training at scale.
- Slightly different behavior: pre-norm models are not strictly equivalent to post-norm.

### Why This Matters for L5

- **Training stability**: pre-norm is the industry standard because it enables training very deep models.
- **Gradient flow**: the clean residual path in pre-norm ensures gradients don't vanish/explode.
- **If asked "how would you modify a transformer for stable training at scale?"**: pre-norm is one of the first answers.

---

## SwiGLU / GeGLU (Modern FFN Activations)

### Standard FFN

```
FFN(x) = ReLU(x @ W1) @ W2
```

or with GELU:

```
FFN(x) = GELU(x @ W1) @ W2
```

### SwiGLU (Used in LLaMA, PaLM)

```
SwiGLU(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
```

Where:
- `Swish(x) = x * sigmoid(x)` (also called SiLU)
- `W_gate` and `W_up` are both `(D, D_ff)` — so there are **three** weight matrices instead of two
- The gating mechanism allows the model to selectively pass information

### GeGLU (Used in some models)

Same idea but with GELU instead of Swish:

```
GeGLU(x) = (GELU(x @ W_gate) * (x @ W_up)) @ W_down
```

### Why Gated Activations Work

- The multiplicative gate lets the model **selectively suppress** certain features.
- Empirically better than ReLU/GELU on language tasks.
- Cost: 50% more FFN parameters (3 matrices instead of 2).
- In practice, `D_ff` is often reduced to compensate (e.g., `2/3 * 4D` instead of `4D`).

### FFN Expansion Ratio

| Model | Expansion | Activation |
|---|---|---|
| Original Transformer | 4x | ReLU |
| GPT-2 | 4x | GELU |
| LLaMA | ~2.67x (8/3) | SwiGLU |
| PaLM | 4x | SwiGLU |

The `8/3` ratio for LLaMA compensates for the extra gate matrix — total params ≈ `3 * (8/3) = 8` vs standard `2 * 4 = 8`.

---

## Sliding Window Attention

### Core Idea

Instead of attending to all N tokens, each token only attends to a window of `W` previous tokens:

```
Attention(i, j) = valid only if |i - j| <= W
```

### Benefits

- **O(N * W)** instead of O(N²) — linear in sequence length for fixed window.
- **Bounded memory**: attention matrix is `(N, W)` not `(N, N)`.
- Used in: Mistral (W=4096), Longformer, Gemma.

### Stacking Layers = Effective Receptive Field Grows

With `L` layers of sliding window attention, the effective receptive field is `L * W`:

```
Layer 1: token sees W tokens back
Layer 2: token sees W tokens back, each of which saw W tokens back
...
Layer L: effective receptive field = L * W
```

So a 32-layer model with W=4096 has an effective receptive field of 131,072 tokens.

### Tradeoffs

| | Full Attention | Sliding Window |
|---|---|---|
| Complexity | O(N²) | O(N * W) |
| Long-range | Direct | Indirect (through layers) |
| Memory | O(N²) | O(N * W) |
| Quality | Best | Slightly worse for very long-range |

---

## FlashAttention-2 Improvements

### What Changed from FlashAttention v1

1. **Better work partitioning**: v1 had uneven GPU thread block utilization. v2 reduces non-MATM FLOPs and better balances work across thread blocks.
2. **Reduced synchronization**: fewer barriers between GPU thread blocks.
3. **Better backward pass**: recomputes attention in backward with optimal tiling.
4. **~2x speedup** over v1 on typical sequence lengths.

### Key Numbers

| | Standard | FlashAttention v1 | FlashAttention v2 |
|---|---|---|---|
| HBM reads/writes | O(N²) | O(N² / M) | O(N² / M) |
| Wall-clock speed | 1x | ~2-4x | ~5-8x |
| Space | O(N²) | O(N) | O(N) |

Where `M` = SRAM size. The speedup comes from reduced memory traffic, not fewer FLOPs.

---

## L5 Interview Q&A

### Q: "Why does attention need positional encoding at all?"

Attention computes `softmax(QKᵀ)V`. The dot product `Q · K` has no notion of position — it's purely content-based. Without positional encoding, "dog bites man" and "man bites dog" produce identical attention patterns because the same tokens attend to the same tokens regardless of order.

### Q: "Compare sinusoidal, learned, RoPE, and ALiBi positional encodings"

| | Sinusoidal | Learned | RoPE | ALiBi |
|---|---|---|---|---|
| Type | Absolute | Absolute | Relative | Relative |
| Extrapolation | Poor | None | Good (with scaling) | Excellent |
| Parameters | 0 | N * D | 0 | 0 |
| Used in | Original Transformer | GPT-2 | LLaMA | BLOOM |
| Mechanism | Add to embeddings | Add to embeddings | Rotate Q/K | Bias on scores |

### Q: "Why is pre-norm more stable than post-norm?"

In post-norm, the residual path goes through LayerNorm: `x = LN(x + f(x))`. This means gradients on the residual path are modulated by the LayerNorm, which can cause vanishing/exploding gradients in deep models.

In pre-norm: `x = x + f(LN(x))`. The residual path is clean — gradients flow directly through the addition. The LayerNorm only normalizes the input to the sublayer, not the gradient highway.

### Q: "What's the effective receptive field of a 32-layer sliding window model with W=4096?"

`32 * 4096 = 131,072` tokens. Each layer extends the receptive field by W. This is why sliding window attention can handle long contexts despite each layer only attending to a local window.

### Q: "When would you use cross attention vs self attention?"

- **Self attention**: when the model should attend within the same sequence (GPT, BERT).
- **Cross attention**: when one sequence should attend to another (encoder-decoder translation, multimodal models like LLaVA where text queries attend to image features).
- **In practice**: decoder-only LLMs (GPT, LLaMA) use only self-attention. Encoder-decoder models (T5, BART) use both.

### Q: "How does SwiGLU compare to ReLU in the FFN?"

SwiGLU adds a gating mechanism: `Swish(xW_gate) * (xW_up)`. The gate allows selective suppression of features, which empirically improves language modeling. The cost is 50% more FFN parameters (3 matrices vs 2), but this is typically compensated by reducing the expansion ratio from 4x to ~2.67x.

---

## Interview Sound Bites

- Attention alone cannot model order — positional encoding fixes this.
- Modern LLMs use RoPE (LLaMA/Gemma/Mistral) or ALiBi (BLOOM) — not sinusoidal.
- RoPE rotates Q/K vectors; ALiBi adds a linear distance bias to attention scores.
- Pre-norm > post-norm for training stability at scale — the residual path must be clean.
- SwiGLU (gated FFN) is the modern replacement for ReLU/GELU — at the cost of one extra weight matrix.
- Sliding window attention gives O(N*W) complexity with effective receptive field = L * W.
- FlashAttention-2 is 5-8x faster than standard attention with the same mathematical result.
- Cross attention lets one sequence ask questions over another — used in encoder-decoder and multimodal models.
- GPT uses **masked self-attention** only — no encoder, no cross attention.
