# KV Cache Implementation

## Goal

During decoding, reuse previous keys/values so each step computes attention only for the new token.

## Without KV Cache

At time `t`, recompute K/V for all `t` tokens.

Per-step work grows with sequence length.

## With KV Cache

Store previous K/V and append only new K/V.

Per-step computation becomes much cheaper.

## Tensor Shapes

For one layer:

- `K_cache`: `(B, H_kv, T, d)`
- `V_cache`: `(B, H_kv, T, d)`

At step `t`, append `k_t`, `v_t` with shape `(B, H_kv, 1, d)`.

## Minimal Pseudocode

```python
def decode_step(x_t, cache, Wq, Wk, Wv):
    # x_t: (B, 1, D)
    q_t = project_q(x_t, Wq)                # (B, H, 1, d)
    k_t = project_k(x_t, Wk)                # (B, H_kv, 1, d)
    v_t = project_v(x_t, Wv)                # (B, H_kv, 1, d)

    # append to cache
    cache['K'] = np.concatenate([cache['K'], k_t], axis=2)
    cache['V'] = np.concatenate([cache['V'], v_t], axis=2)

    # attention of current query against all cached keys
    out_t = attention(q_t, cache['K'], cache['V'])
    return out_t, cache
```

## Causal Mask in Decoding

For one-token decode, causal masking is implicit because query length is 1 and keys only include past+current tokens.

## Common Engineering Notes

- Pre-allocate cache for max length to avoid repeated concat
- Keep cache in contiguous GPU memory
- Quantize KV cache (fp8/int8 variants) for memory reduction

## Complexity Intuition

- No cache: repetitive recomputation of past K/V
- With cache: compute new K/V once, reuse past

## Pre-Allocated Cache (Production Style)

In production, `np.concatenate` is too slow — it allocates new memory each step. Instead, **pre-allocate** the full cache and fill it incrementally:

```python
class KVCache:
    def __init__(self, batch_size, n_layers, n_kv_heads, max_seq_len, head_dim, dtype=np.float16):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        # Pre-allocate: (n_layers, B, H_kv, max_seq_len, d)
        self.K = np.zeros((n_layers, batch_size, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        self.V = np.zeros((n_layers, batch_size, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        self.current_len = 0

    def append(self, layer_idx, k_new, v_new):
        # k_new, v_new: (B, H_kv, 1, d)
        pos = self.current_len
        self.K[layer_idx, :, :, pos:pos+1, :] = k_new
        self.V[layer_idx, :, :, pos:pos+1, :] = v_new

    def get(self, layer_idx):
        # Return only the filled portion
        return (
            self.K[layer_idx, :, :, :self.current_len+1, :],
            self.V[layer_idx, :, :, :self.current_len+1, :]
        )

    def step(self):
        self.current_len += 1
```

### Why Pre-Allocation Matters

- `np.concatenate`: O(N) copy each step → O(N²) total for N tokens.
- Pre-allocated: O(1) write each step → O(N) total.
- In production (vLLM, TensorRT-LLM): uses PagedAttention for even better memory management.

---

## Multi-Layer KV Cache

In a real transformer, **every layer** has its own KV cache:

```python
def generate_with_cache(model, prompt_tokens, max_new_tokens):
    # Initialize cache for all layers
    cache = KVCache(batch_size=1, n_layers=model.n_layers,
                    n_kv_heads=model.n_kv_heads,
                    max_seq_len=model.max_seq_len,
                    head_dim=model.head_dim)

    # Prefill: process all prompt tokens
    for layer_idx in range(model.n_layers):
        k, v = model.compute_kv(layer_idx, prompt_tokens)
        cache.K[layer_idx] = k  # (B, H_kv, N_prompt, d)
        cache.V[layer_idx] = v

    cache.current_len = len(prompt_tokens)

    # Decode: generate one token at a time
    generated = []
    for step in range(max_new_tokens):
        token = generated[-1] if generated else prompt_tokens[-1]

        for layer_idx in range(model.n_layers):
            k_new, v_new = model.compute_kv(layer_idx, token)  # (B, H_kv, 1, d)
            cache.append(layer_idx, k_new, v_new)

            k_all, v_all = cache.get(layer_idx)
            q = model.compute_q(layer_idx, token)  # (B, H, 1, d)
            out = attention(q, k_all, v_all)       # (B, H, 1, d)

        cache.step()
        next_token = model.sample(out)
        generated.append(next_token)

    return generated
```

### Key Insight

The KV cache is **per-layer** — each of the 32 layers (in a 7B model) has its own K and V cache. The total memory is:

```
Total KV cache = 2 * n_layers * B * H_kv * N * d * bytes_per_param
```

---

## GQA-Aware KV Cache

With GQA, the cache only stores `H_kv` heads (not `H` query heads):

```python
class GQAKVCache:
    def __init__(self, batch_size, n_layers, n_q_heads, n_kv_heads, max_seq_len, head_dim):
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads

        # Only store n_kv_heads, not n_q_heads
        self.K = np.zeros((n_layers, batch_size, n_kv_heads, max_seq_len, head_dim))
        self.V = np.zeros((n_layers, batch_size, n_kv_heads, max_seq_len, head_dim))

    def get_for_query_head(self, layer_idx, q_head_idx):
        """Map query head to its KV group."""
        kv_head_idx = q_head_idx // self.group_size
        return self.K[layer_idx, :, kv_head_idx, ...], self.V[layer_idx, :, kv_head_idx, ...]
```

### Memory Savings

For LLaMA-2 7B (32 Q heads, 8 KV heads):
- MHA cache: `2 * 32 * 32 * 4096 * 128 * 2 = 2 GB`
- GQA cache: `2 * 32 * 8 * 4096 * 128 * 2 = 512 MB` (4x savings)

---

## Prefix Caching

### The Idea

Many requests share the same **system prompt** or **few-shot examples**. The KV cache for this prefix can be **shared** across requests.

```
Request 1: [System prompt] [User query 1]
Request 2: [System prompt] [User query 2]
Request 3: [System prompt] [User query 3]
```

The KV cache for `[System prompt]` is computed once and reused.

### Implementation

```python
class PrefixCache:
    def __init__(self):
        self.cache = {}  # token_hash -> KV blocks

    def get_or_compute(self, tokens, model):
        # Find longest cached prefix
        prefix_len = 0
        for i in range(len(tokens), 0, -1):
            h = hash(tuple(tokens[:i]))
            if h in self.cache:
                prefix_len = i
                break

        # Reuse cached prefix
        if prefix_len > 0:
            cached_kv = self.cache[hash(tuple(tokens[:prefix_len]))]
            # Only compute KV for new tokens
            new_kv = model.compute_kv(tokens[prefix_len:])
            return concat(cached_kv, new_kv)
        else:
            return model.compute_kv(tokens)
```

### Impact

- **Chatbots**: system prompt is often 500-2000 tokens. Caching it saves significant prefill compute.
- **Few-shot**: shared examples across requests.
- **vLLM**: implements this via PagedAttention's copy-on-write mechanism.

---

## KV Cache Quantization

### FP8 / INT8 KV Cache

Reduce KV cache memory by 50% (FP16 → INT8) or 75% (FP16 → INT4):

```python
def quantize_kv(k, v, scale_k, scale_v):
    k_int8 = np.round(k / scale_k).astype(np.int8)
    v_int8 = np.round(v / scale_v).astype(np.int8)
    return k_int8, v_int8

def dequantize_kv(k_int8, v_int8, scale_k, scale_v):
    return k_int8 * scale_k, v_int8 * scale_v
```

### Tradeoffs

| Precision | Memory | Quality Impact | Speed |
|---|---|---|---|
| FP16 | 1x | None | Baseline |
| FP8 | 0.5x | Minimal | Faster (less memory traffic) |
| INT8 | 0.5x | Small | Faster |
| INT4 | 0.25x | Moderate | Fastest but risky |

### When to Use

- **Long context** (128K+): KV cache dominates memory → quantization is essential.
- **High batch size**: more requests = more KV cache = more benefit from quantization.
- **Quality-sensitive**: use FP8 (minimal quality impact).

---

## L5 Interview Q&A

### Q: "Walk through the complete memory budget for serving a 7B model with 100 concurrent users at 4K context."

```
Model weights (FP16):     7B * 2 bytes = 14 GB
KV cache (GQA, 8 heads):
  Per request: 2 * 32 * 8 * 4096 * 128 * 2 = 512 MB
  100 requests: 512 MB * 100 = 51.2 GB
Activations: ~2 GB (small during decode)
Overhead: ~2 GB

Total: ~69 GB → fits on a single 80GB A100
```

Without GQA (MHA, 32 heads): KV cache = 2 GB * 100 = 200 GB → needs 3-4 GPUs just for KV cache.

### Q: "How would you implement KV cache for a multi-GPU serving setup?"

With tensor parallelism (TP=8):
1. Each GPU holds `n_kv_heads / TP` KV heads.
2. With GQA (8 KV heads) and TP=8: each GPU has 1 KV head.
3. Each GPU stores its portion of K/V cache locally.
4. After attention, each GPU has partial results → all-reduce to combine.
5. No need to communicate KV cache across GPUs (each GPU's Q attends to local K/V).

### Q: "What happens if the KV cache exceeds GPU memory?"

1. **OOM**: the request fails (worst case).
2. **Offloading**: move some KV cache to CPU memory (slower but works).
3. **Eviction**: drop old KV entries (sliding window) — degrades quality for long-range attention.
4. **Quantization**: compress KV cache to INT8/FP8 to fit more.
5. **PagedAttention**: better memory utilization may prevent OOM in the first place.

### Q: "How does prefix caching interact with PagedAttention?"

In PagedAttention, KV cache is stored in blocks. Prefix caching becomes natural:
1. Compute KV for the shared prefix → store in blocks.
2. New requests with the same prefix: **reference** the same blocks (copy-on-write).
3. When a request modifies a block (new tokens), copy it first (COW).
4. No re-computation for the shared prefix.

This is how vLLM achieves prefix caching efficiently.

### Q: "Why is the KV cache stored per-layer and not globally?"

Each transformer layer has **different** K and V projections (`W_k`, `W_v` are layer-specific). The K/V at layer 0 are completely different from K/V at layer 31. They can't be shared or merged. Each layer's attention only uses its own K/V.

---

## Interview Sound Bites

- KV cache stores K/V per layer, not Q — new token gets new Q, so attention scores are recomputed each step.
- Pre-allocate cache to avoid O(N²) concat overhead — use O(1) writes into pre-allocated buffer.
- With GQA, only store `n_kv_heads` in cache — 4x memory savings for LLaMA-2 7B.
- **Prefix caching**: share KV for common system prompts — major TTFT reduction for chatbots.
- **KV cache quantization** (FP8/INT8): 50-75% memory reduction with minimal quality impact.
- Total KV cache memory: `2 * n_layers * B * n_kv_heads * N * d * bytes_per_param`.
- PagedAttention + prefix caching = copy-on-write block sharing for maximum memory efficiency.
- In tensor parallelism, each GPU stores only its share of KV heads — no cross-GPU KV communication.
