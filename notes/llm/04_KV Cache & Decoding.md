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

## Implementation Priority for Interviews

| Priority | Strategy |
|---|---|
| Must implement | Greedy, Temperature, Top-K |
| Good to know | Top-P |
| Conceptual only | Beam Search |
