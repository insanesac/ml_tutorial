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

## Interview Sound Bites

- Attention alone cannot model order — positional encoding fixes this.
- Cross attention lets one sequence ask questions over another.
- Decoder queries (`Q`), encoder provides information (`K`, `V`).
- GPT uses **masked self-attention** only — no encoder, no cross attention.
- N_dec and N_enc are independent — cross attention score matrix is rectangular.

---

## Flash Attention

### The Problem with Standard Attention

Standard attention materializes the full `(N, N)` score matrix in GPU HBM (high-bandwidth memory):

```
scores = Q @ Kᵀ        # (N, N)  <- written to HBM
A = softmax(scores)    # (N, N)  <- written to HBM
out = A @ V
```

For long sequences this is expensive in **memory**, not compute.

### Complexity Comparison

| | Time | Space |
|---|---|---|
| Standard Attention | O(N²) | O(N²) |
| Flash Attention | O(N²) | O(N) |

Same time complexity. Flash Attention's win is entirely **space**.

### The Insight

Never materialize the full N×N matrix.

Instead, tile Q, K, V into blocks that fit in fast on-chip SRAM. Compute partial softmax results per block, accumulate the output incrementally, discard the block once done.

The full N×N matrix is **never written to HBM**.

### Why Softmax Is the Hard Part

Softmax needs the full row to normalize:

```
A_i = exp(z_i) / Σ exp(z_j)
```

Flash Attention uses the **online softmax trick** — maintain a running max and running sum as you scan blocks, then correct at the end. This lets softmax be computed without seeing all values at once.

### Why It Matters

- Longer context windows become feasible.
- Memory cost does not explode with sequence length.
- GPU memory is usually the bottleneck, not FLOPS.

### Interview Sound Bite

> Flash Attention has the same O(N²) time complexity as standard attention but reduces space from O(N²) to O(N) by tiling computation into blocks and using the online softmax trick — the full attention matrix is never materialized in memory.
