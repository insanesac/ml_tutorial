# GQA & MQA (Grouped / Multi-Query Attention)

## Motivation

In autoregressive decoding, KV cache dominates memory.

Standard MHA stores separate K/V for every head:

`KV cache ~ O(H * N * D_head)`

GQA and MQA reduce KV memory.

## Standard MHA

- Query heads: `H`
- Key heads: `H`
- Value heads: `H`

Highest quality, highest KV memory.

## MQA (Multi-Query Attention)

- Query heads: `H`
- Key heads: `1`
- Value heads: `1`

All query heads share same K and V.

### Pros
- Big KV cache reduction
- Faster decoding

### Cons
- Can hurt quality vs full MHA

## GQA (Grouped-Query Attention)

Middle ground between MHA and MQA.

- Query heads: `H`
- Key/Value heads: `G` where `1 < G < H`

Each group of query heads shares one K/V head.

## KV Memory Comparison

For fixed `N, D_head`:

- MHA: proportional to `H`
- GQA: proportional to `G`
- MQA: proportional to `1`

So `MQA < GQA < MHA` in memory.

## Why GQA Is Popular

It gives most of MQA’s speed/memory gain with less quality drop.

This is why many modern serving stacks prefer GQA.

## Implementation Intuition

1. Compute Q with `H` heads as usual.
2. Compute K/V with `G` heads.
3. Map each query head to its KV group.
4. Attention per query head uses grouped K/V.

## Concrete KV Cache Memory Math (L5 Essential)

### Setup

```
Model: LLaMA-2 7B
Layers: 32
Query heads (H): 32
KV heads: 32 (MHA), 8 (GQA), 1 (MQA)
Head dim (d): 128
Seq length (N): 4096
Batch (B): 1
Precision: FP16 (2 bytes)
```

### MHA (32 KV heads)

```
KV cache = 2 (K+V) * 32 (layers) * 32 (heads) * 4096 (tokens) * 128 (dim) * 2 (bytes)
         = 2 * 32 * 32 * 4096 * 128 * 2
         = 2,147,483,648 bytes = 2 GB per request
```

### GQA (8 KV heads)

```
KV cache = 2 * 32 * 8 * 4096 * 128 * 2
         = 536,870,912 bytes = 512 MB per request
```

### MQA (1 KV head)

```
KV cache = 2 * 32 * 1 * 4096 * 128 * 2
         = 67,108,864 bytes = 64 MB per request
```

### At Batch Size 64

| Method | KV Cache per Request | Total KV Cache (64 reqs) |
|---|---|---|
| MHA | 2 GB | 128 GB |
| GQA | 512 MB | 32 GB |
| MQA | 64 MB | 4 GB |

**This is why GQA is critical for serving** — at batch size 64, MHA needs 128 GB just for KV cache, while GQA needs only 32 GB.

---

## Implementation (NumPy)

```python
def gqa_attention(Q, K, V, n_kv_heads, n_q_heads):
    """
    Q: (B, n_q_heads, N, d)     — query heads
    K: (B, n_kv_heads, N, d)    — KV heads (fewer)
    V: (B, n_kv_heads, N, d)
    n_kv_heads: number of KV heads (G)
    n_q_heads: number of query heads (H)
    """
    B, H, N, d = Q.shape
    G = n_kv_heads
    group_size = H // G  # how many Q heads share one KV head

    # Repeat K, V to match Q heads
    K_expanded = np.repeat(K, group_size, axis=1)  # (B, H, N, d)
    V_expanded = np.repeat(V, group_size, axis=1)  # (B, H, N, d)

    # Standard attention
    scores = Q @ K_expanded.transpose(0, 1, 3, 2) / np.sqrt(d)  # (B, H, N, N)
    weights = softmax(scores, axis=-1)
    return weights @ V_expanded  # (B, H, N, d)
```

### Key Implementation Detail

The `repeat` operation is conceptual — in practice, you don't actually copy K/V. Instead, you use **index mapping** or **broadcasting** to avoid materializing the expanded tensors:

```python
# Efficient: use indexing instead of repeat
head_to_kv = torch.arange(H) // group_size  # maps Q head -> KV head
K_grouped = K[:, head_to_kv, :, :]  # (B, H, N, d) — view, not copy
```

---

## Quality Impact

### Benchmark Results (from the GQA paper, Ainslie et al. 2023)

| Method | KV Heads | Quality (vs MHA) | Speedup |
|---|---|---|---|
| MHA | 32 | 100% (baseline) | 1x |
| GQA-8 | 8 | ~99% | ~1.5-2x |
| GQA-4 | 4 | ~97-98% | ~2-3x |
| MQA | 1 | ~92-95% | ~3-4x |

### Key Takeaway

- **GQA with 8 groups**: nearly indistinguishable quality from MHA, significant memory savings.
- **MQA**: noticeable quality drop, but maximum memory savings.
- **The sweet spot is GQA** — this is why LLaMA-2 70B, Mistral, and Gemma all use it.

### Models Using GQA

| Model | Query Heads | KV Heads | Group Size |
|---|---|---|---|
| LLaMA-2 70B | 64 | 8 | 8 |
| Mistral 7B | 32 | 8 | 4 |
| Gemma 7B | 16 | 8 | 2 |
| LLaMA-3 8B | 32 | 8 | 4 |

---

## Why GQA Doesn't Hurt Quality Much

### Intuition

- Different attention heads learn different patterns (syntactic, semantic, positional).
- But many heads are **redundant** — they learn similar things.
- Sharing K/V across heads forces the shared KV to be more general, which acts as a mild regularizer.
- The Q heads still differ, so they can attend to different aspects of the same K/V representations.

### The "Redundant Heads" Finding

Research (Voita et al., 2019) showed that many attention heads can be pruned without quality loss. GQA exploits this — instead of pruning heads, it shares their K/V projections.

---

## Training vs Inference Impact

| | Training | Inference (Prefill) | Inference (Decode) |
|---|---|---|---|
| MHA | Full compute | Full compute | Full KV cache |
| GQA | Slightly less (fewer KV projections) | Slightly less | **Much less KV cache** |
| MQA | Less (1 KV projection) | Less | **Minimal KV cache** |

The main win is in **decode phase** where KV cache dominates memory. During training and prefill, the compute savings are smaller because Q projection still dominates.

---

## L5 Interview Q&A

### Q: "How does GQA affect the KV cache memory for a 70B model serving 100 concurrent users?"

For LLaMA-2 70B (64 Q heads, 8 KV heads, 80 layers, d=128, 4K context, FP16):

```
MHA:  2 * 80 * 64 * 4096 * 128 * 2 * 100 = 5.12 TB  (infeasible)
GQA:  2 * 80 *  8 * 4096 * 128 * 2 * 100 = 640 GB    (feasible with multiple GPUs)
MQA:  2 * 80 *  1 * 4096 * 128 * 2 * 100 = 80 GB     (very feasible)
```

GQA makes serving 70B at scale possible; MHA does not.

### Q: "Why not always use MQA?"

MQA's quality drop (~5-8%) is significant for production models. The shared K/V becomes a bottleneck — all 32 query heads must work with a single K/V representation, which limits the model's ability to attend to different aspects of the input.

GQA gives ~90% of the memory savings with negligible quality loss. It's the better tradeoff for most applications.

### Q: "How do you convert an MHA model to GQA post-hoc?"

1. **Merge K/V heads**: average the K/V projection weights for heads in the same group.
2. **Fine-tune**: short fine-tuning run to recover any quality loss from the merging.
3. This is how some MHA models are converted to GQA for efficient serving.

### Q: "Does GQA affect training speed?"

Minimally. The Q projection (which dominates attention compute) is unchanged. The K/V projections are smaller, but they're a small fraction of total compute. The main benefit is inference memory, not training speed.

### Q: "What's the relationship between GQA and tensor parallelism?"

GQA actually **helps** tensor parallelism. With TP, K/V heads are split across GPUs. With MHA (32 heads) and TP=8, each GPU gets 4 KV heads. With GQA (8 heads) and TP=8, each GPU gets 1 KV head — simpler and more memory-efficient.

But: if `n_kv_heads < TP_size`, you can't split evenly. GQA-4 with TP=8 requires careful handling (some GPUs have 0 KV heads, which needs special attention).

---

## Interview Sound Bites

- GQA/MQA are decoding-time optimizations via smaller KV cache — they reduce K/V heads, not Q heads.
- MQA = extreme sharing (1 KV head), GQA = partial sharing (G groups).
- GQA gives ~90% of MQA's memory savings with negligible quality loss — this is why it's the industry standard.
- KV cache math: `2 * layers * n_kv_heads * seq_len * head_dim * bytes * batch_size`.
- For a 70B model at batch 100: MHA needs 5TB, GQA needs 640GB, MQA needs 80GB.
- GQA acts as mild regularization — redundant heads share K/V, forcing more general representations.
- Used in LLaMA-2/3, Mistral, Gemma — essentially all modern LLMs.
- GQA helps tensor parallelism: fewer KV heads = simpler sharding.
