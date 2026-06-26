# KV Cache & Decoding

## KV Cache

### The Problem

During autoregressive generation, previously generated tokens do not change.

Without a cache, every new token forces recomputation of **all** K and V matrices from the start.

### The Insight

Store K and V from previous tokens. On each new step, compute only the new token's K and V.

**Example: generating "developer" after "I am a great"**

| Without KV Cache | With KV Cache |
|---|---|
| Recompute K1..K5, V1..V5 | Reuse K1..K4, V1..V4 |
| O(t²) per token | O(t) per token |

### What Is / Is Not Cached

| Cached | Not Cached |
|---|---|
| Keys (K) | Queries (Q) |
| Values (V) | Attention scores |

Why no cached attention scores?

Attention scores depend on the current query. New token = new Q = new scores.

### Complexity

| Mode | Per-token cost |
|---|---|
| No cache | O(t²) |
| With cache | O(t) |

### Tradeoff

More memory. Less compute.

### Interview Sound Bite

> KV Cache trades memory for speed by storing previously computed keys and values, reducing per-token attention cost from O(t²) to O(t).

---

## Decoding Strategies

### Greedy Decoding

Always pick the highest probability token.

```python
next_token = np.argmax(probs)
```

- **Complexity:** O(V) where V = vocab size
- **Pro:** Fast, deterministic.
- **Con:** Can miss globally better sequences.

---

### Temperature Scaling

Applied **before** softmax:

```python
probs = softmax(logits / T)
```

| Temperature | Effect |
|---|---|
| T < 1 | Sharper — more deterministic |
| T = 1 | No change |
| T > 1 | Flatter — more random |
| T → 0 | Approaches argmax |
| T → ∞ | Approaches uniform distribution |

---

### Top-K Sampling

Keep only the top K tokens, renormalize, then sample.

```python
idx = np.argsort(probs)[-k:]
top_probs = probs[idx]
top_probs /= top_probs.sum()
next_token = np.random.choice(idx, p=top_probs)
```

- **Pro:** Simple.
- **Con:** Always keeps exactly K candidates regardless of distribution shape.

---

### Top-P (Nucleus) Sampling

Keep tokens until cumulative probability exceeds threshold P, then renormalize and sample.

**Example with P = 0.9:**

```
Token A: 0.40  -> cumsum 0.40
Token B: 0.30  -> cumsum 0.70
Token C: 0.20  -> cumsum 0.90  <- stop here
```

- **Pro:** Adaptive — wide distribution gets more candidates, peaked distribution gets fewer.
- **Interview Sound Bite:** Top-P adapts to model confidence while Top-K uses a fixed count.

```python
def top_p_sampling(probs, p):
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1

    top_p_indices = sorted_indices[:cutoff]
    top_p_probs = probs[top_p_indices]
    top_p_probs = top_p_probs / top_p_probs.sum()

    return np.random.choice(top_p_indices, p=top_p_probs)
```

---

### Beam Search

Maintain multiple candidate sequences (beams) simultaneously.

**Example with beam width = 2:**

```
Step 1: "The cat sat", "The cat slept"
Step 2: expand both, keep top 2 by score
```

- **Pro:** Better global sequence quality.
- **Con:** Slower and more memory-intensive.

---

## Comparison

| Strategy | Speed | Diversity | Complexity |
|---|---|---|---|
| Greedy | Fastest | None | O(V) |
| Temperature | Fast | Moderate | O(V) |
| Top-K | Fast | Controlled | O(V + K log K) |
| Top-P | Fast | Adaptive | O(V log V) |
| Beam Search | Slow | Low | O(beam_width * V) |

## Prefill vs Decode Phases (L5 Critical)

### Prefill Phase

Processing the **prompt** (input tokens) before generation begins.

```
Input: "Explain quantum computing" (4 tokens)
Prefill: compute K, V for all 4 tokens simultaneously
         → O(N²) attention (full matrix)
         → Compute-bound (good GPU utilization)
```

- **Parallel**: all prompt tokens processed at once.
- **Compute-bound**: large matrix multiplications, GPU is well-utilized.
- **FlashAttention helps most here**: avoids materializing N×N matrix.

### Decode Phase

Generating **one token at a time** autoregressively.

```
Step 1: generate token 5 (using KV cache for tokens 1-4)
Step 2: generate token 6 (using KV cache for tokens 1-5)
...
```

- **Sequential**: one token at a time.
- **Memory-bound**: small matrix multiplications, GPU underutilized.
- **KV cache essential**: avoids recomputing all previous K/V.
- **Batching is key**: process multiple requests simultaneously to keep GPU busy.

### Why This Matters for Serving

```
TTFT (Time To First Token) ≈ prefill time
TPOT (Time Per Output Token) ≈ decode time per token
```

- **TTFT** is dominated by prefill — proportional to prompt length.
- **TPOT** is dominated by memory bandwidth — each decode step loads weights + KV cache.
- **Optimization**: prefill is compute-bound (use FlashAttention), decode is memory-bound (use batching, GQA, quantization).

---

## PagedAttention (vLLM)

### The Problem with Naive KV Cache

Each request pre-allocates KV cache for its **maximum** sequence length:

```
Request 1: 2048 tokens allocated, uses 100  → 95% wasted
Request 2: 2048 tokens allocated, uses 2000 → 2% wasted
```

This causes severe **memory fragmentation** — you can't pack more requests because memory is reserved but unused.

### PagedAttention Solution

Borrow from OS virtual memory:

1. Divide KV cache into fixed-size **blocks** (e.g., 16 tokens per block).
2. Maintain a **block table** mapping logical positions to physical blocks.
3. Allocate blocks **on demand** — only as many as needed.
4. Free blocks when a request finishes.

```
Request 1: [Block 0] [Block 5] [Block 3]    → 3 blocks, 48 tokens
Request 2: [Block 1] [Block 2] [Block 7] [Block 8] → 4 blocks, 64 tokens
```

### Benefits

- **No fragmentation**: blocks are allocated dynamically.
- **Higher batch sizes**: more requests fit in the same memory.
- **Prefix sharing**: multiple requests with the same system prompt share KV blocks (copy-on-write).
- **2-4x throughput** improvement over naive allocation.

### Why This Is a Game Changer for L5

PagedAttention is the core innovation behind vLLM. It enables:
- Serving 10-100x more concurrent requests on the same hardware.
- Dynamic batching without pre-allocating memory.
- Memory utilization going from ~20% to ~90%+.

---

## Speculative Decoding

### The Problem

During decode, each step generates one token — the GPU is **memory-bound** (loading weights for one token's worth of compute). Most of the GPU's compute capacity is idle.

### The Solution

Use a **small draft model** to generate `k` candidate tokens cheaply, then verify them with the **large model** in a single forward pass:

```
1. Draft model generates tokens [t1, t2, t3] (fast, 3 sequential steps)
2. Large model verifies all 3 in one forward pass (parallel)
3. Accept tokens that match, reject at first mismatch
4. Accepted tokens are added to output; rejected token is replaced
```

### Why It's Faster

- Large model processes 3 tokens in **one forward pass** instead of 3 sequential passes.
- If the draft model is good, 2-3 tokens are accepted per verification → ~2-3x speedup.
- The large model's forward pass for 3 tokens costs about the same as for 1 token (memory-bound).

### Key Constraint

**The output distribution must be identical** to autoregressive decoding. This is ensured by:
- Accepting draft tokens with probability `min(1, p_large / p_draft)`.
- If rejected, sampling from the residual distribution `(p_large - p_draft)_+`.

This guarantees the output is **exactly** the same distribution as standard decoding — no quality loss.

### Draft Model Selection

| Strategy | Example |
|---|---|
| Smaller version of same model | LLaMA-7B as draft for LLaMA-70B |
| Same model, fewer layers | Early exit from same model |
| N-gram model | No neural network — just lookup |
| Medusa heads | Multiple prediction heads on same model |

---

## Advanced Decoding Parameters

### Repetition Penalty

Reduces probability of tokens that have already appeared:

```python
for token_id in generated_tokens:
    logits[token_id] /= penalty  # penalty > 1, e.g., 1.1-1.3
```

- **Effect**: discourages repeating the same token.
- **Typical value**: 1.1 - 1.3.
- **Problem**: also penalizes legitimate repetition (e.g., "the the the" in a list).

### Frequency Penalty

Penalizes tokens proportional to how many times they've appeared:

```python
for token_id, count in token_counts.items():
    logits[token_id] -= count * penalty
```

- **Effect**: stronger penalty for frequently repeated tokens.
- **Typical value**: 0.1 - 1.0.

### Presence Penalty

Binary penalty — reduces logit by a fixed amount if token has appeared at all:

```python
for token_id in set(generated_tokens):
    logits[token_id] -= penalty
```

- **Effect**: encourages topical diversity.
- **Typical value**: 0.1 - 1.0.

### Comparison

| Penalty | Target | Mechanism |
|---|---|---|
| Repetition | Exact token repetition | Divides logits |
| Frequency | Frequent tokens | Subtracts proportional to count |
| Presence | Token diversity | Subtracts fixed amount |

---

## Constrained Decoding

### JSON Mode

Force the model to output valid JSON:

```
1. Define a grammar (JSON schema)
2. At each step, mask logits to only allow valid next tokens
3. Model can only produce tokens that lead to valid JSON
```

### How It Works

```python
def constrained_decode(logits, grammar_state):
    valid_tokens = grammar_state.get_valid_next_tokens()
    mask = np.full_like(logits, -inf)
    mask[valid_tokens] = 0
    return softmax(logits + mask)
```

- **Guarantee**: output is always valid per the grammar.
- **Cost**: grammar tracking overhead, slightly reduced diversity.
- **Used in**: OpenAI structured outputs, guidance, outlines.

---

## L5 Interview Q&A

### Q: "How would you reduce TTFT for a chatbot?"

1. **Prefix caching**: cache KV for common system prompts.
2. **Speculative decoding**: start generating while still processing prompt (overlap).
3. **Smaller model for prefill**: use a smaller model for the first token, switch to large model.
4. **Chunked prefill**: process prompt in chunks to allow interleaving with decode steps.
5. **FlashAttention**: faster prefill computation.

### Q: "How does continuous batching work?"

In traditional batching, all requests in a batch must finish before any new request starts. In continuous batching:

1. Each request is processed **token-by-token** independently.
2. When a request finishes (EOS or max length), it's removed from the batch.
3. A new request is immediately inserted into the freed slot.
4. No waiting for batch completion.

This requires:
- **PagedAttention**: dynamic memory allocation for KV cache.
- **Iteration-level scheduling**: re-evaluate batch composition every token.
- **Result**: 10-100x better throughput vs static batching.

### Q: "What's the memory breakdown during LLM serving?"

```
Model weights:     ~70% of GPU memory (e.g., 14GB for 7B FP16)
KV cache:          ~20-30% (grows with batch size and sequence length)
Activations:       ~5% (small during decode, one token at a time)
Overhead:          ~5% (CUDA context, framework, etc.)
```

For a 7B model on an 80GB A100:
- Weights: 14GB
- Available for KV cache: ~60GB
- With GQA (8 KV heads): ~60GB / 512MB per request = ~120 concurrent requests

### Q: "When is beam search better than sampling?"

- **Beam search**: when you need the **optimal** sequence (machine translation, summarization, code generation). Deterministic, finds high-probability sequences.
- **Sampling**: when you need **diversity** (creative writing, dialogue, brainstorming). Stochastic, produces varied outputs.
- **Modern trend**: most chatbots use sampling (temperature + top-p) because beam search produces repetitive, generic responses in open-ended generation.

### Q: "How does speculative decoding maintain output quality?"

The acceptance/rejection mechanism ensures the output distribution is **mathematically identical** to standard autoregressive decoding. The draft model only proposes tokens — the large model decides whether to accept them. If a draft token has higher probability under the large model, it's always accepted. If lower, it's accepted with probability `p_large / p_draft`. This is a form of rejection sampling that preserves the exact target distribution.

---

## Implementation Priority for Interviews

| Priority | Strategy |
|---|---|
| Must implement | Greedy, Temperature, Top-K, Top-P |
| Good to know | Beam Search, Repetition Penalty |
| Conceptual only | Speculative Decoding, Constrained Decoding |

---

## Interview Sound Bites

- KV cache trades memory for speed: O(t²) → O(t) per token. Store K/V, not Q or attention scores.
- **Prefill** is compute-bound (process all prompt tokens at once); **decode** is memory-bound (one token at a time).
- TTFT ≈ prefill time; TPOT ≈ decode time per token. Optimize them differently.
- **PagedAttention** (vLLM): OS-style virtual memory for KV cache — eliminates fragmentation, enables 2-4x throughput.
- **Continuous batching**: insert/remove requests at token granularity — 10-100x throughput vs static batching.
- **Speculative decoding**: draft model proposes tokens, large model verifies in one pass — 2-3x speedup, zero quality loss.
- Repetition penalty divides logits; frequency/presence penalties subtract from logits.
- Constrained decoding (JSON mode) masks invalid tokens at each step — guarantees valid output.
- Beam search for translation/summarization; sampling for creative/dialogue tasks.
