# Perplexity, LayerNorm & RMSNorm

## Perplexity

### Definition

```
Perplexity = exp(Cross Entropy)
```

### Intuition

Perplexity = **effective number of choices the model is considering per token**.

Lower is better. Minimum possible perplexity is 1.

### Examples

| Cross Entropy | Perplexity | Interpretation |
|---|---|---|
| 0 | 1 | Perfect prediction |
| ln(2) ≈ 0.693 | 2 | Choosing between 2 options |
| ln(4) ≈ 1.386 | 4 | Uniform over 4 choices |

### Implementation

```python
perplexity = np.exp(cross_entropy)
```

### Complexity

- Time: `O(1)` (scalar transformation of CE)
- Space: `O(1)`

### Interview Answer

> Perplexity is exponentiated cross entropy. It is easier to interpret because it corresponds to the effective number of choices the model considers. A GPT-2 class model has perplexity ~30 on standard benchmarks.

---

## LayerNorm

### Purpose

Stabilize activations during training and make optimization landscapes smoother.

### Formula

```
y = gamma * ((x - mean) / sqrt(var + eps)) + beta
```

### Steps

1. Compute mean of `x`
2. Compute variance of `x`
3. Normalize: `x_norm = (x - mean) / sqrt(var + eps)`
4. Scale by learned `gamma`
5. Shift by learned `beta`

### Implementation

```python
def layer_norm(x, eps=1e-5):
    mean = np.mean(x)
    variance = np.mean((x - mean) ** 2)
    return (x - mean) / np.sqrt(variance + eps)
```

### Why epsilon?

Prevents division by zero. Adds numerical stability.

### Why gamma and beta?

Without them, every layer is forced to output mean=0, std=1.

- `gamma` learns the appropriate scale.
- `beta` learns the appropriate shift.

The model can still learn to "undo" normalization if needed.

### Transformer Shape

Input: `(B, N, D)`

LayerNorm normalizes across `D` (embedding dimension).

Each token is normalized **independently** from all other tokens.

### Complexity

`O(B * N * D)`

### Why Not BatchNorm?

| BatchNorm | LayerNorm |
|---|---|
| Normalizes across batch | Normalizes across features |
| Fails at batch size = 1 | Works at batch size = 1 |
| Bad for autoregressive | Good for autoregressive |
| Needs batch statistics at inference | Independent of batch at inference |

---

## RMSNorm

### Purpose

A cheaper alternative to LayerNorm. Used in LLaMA, Gemma, and most modern LLMs.

### Formula

```
x_norm = x / sqrt(mean(x²) + eps)
```

Where `sqrt(mean(x²))` is the Root Mean Square (RMS).

### Key Difference

| | LayerNorm | RMSNorm |
|---|---|---|
| Mean centering | Yes `(x - mean)` | No |
| Scale term | std | RMS |

### Implementation

```python
def rms_norm(x, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms
```

### Why Faster?

LayerNorm requires:
1. `mean(x)`
2. `x - mean`
3. Variance computation

RMSNorm skips all mean-centering. Only one pass over `x` needed.

### Parameters

- Keeps `gamma` (scale)
- Removes `beta` (shift) — empirically not needed

### Complexity

`O(B * N * D)` — same asymptotic, but with a smaller constant.

### Used In

- LLaMA (all versions)
- Gemma
- Mistral
- Most post-2023 open-weight LLMs

### Interview Answer

> RMSNorm normalizes using root mean square instead of full variance and avoids mean-centering entirely, making it computationally cheaper. It retains most benefits of LayerNorm and is the preferred norm in modern LLMs like LLaMA.

---

## All Normalization Layers Compared

| | BatchNorm | LayerNorm | RMSNorm | InstanceNorm | GroupNorm |
|---|---|---|---|---|---|
| Normalize across | Batch | Features | Features (no mean) | Spatial (per channel) | Channel groups |
| Batch dependent | Yes | No | No | No | No |
| Works at batch=1 | No | Yes | Yes | Yes | Yes |
| Used in LLMs | No | Older | Modern (LLaMA) | No | No |
| Parameters | γ, β | γ, β | γ only | γ, β | γ, β |
| Compute | mean + var | mean + var | RMS only | mean + var | mean + var |

### Why LayerNorm/RMSNorm for Transformers (Not BatchNorm)

1. **Batch dependency**: BatchNorm needs batch statistics — fails at batch=1 (inference).
2. **Autoregressive**: variable sequence lengths make batch statistics unreliable.
3. **Padding**: padded positions corrupt batch statistics.
4. **Training-inference mismatch**: BatchNorm needs running averages; LayerNorm doesn't.

---

## Perplexity Deep Dive

### Perplexity and Entropy

```
Perplexity = exp(H(p, q))
           = exp(-Σ p(x) * log(q(x)))
```

Where `p` is the true distribution and `q` is the model's prediction.

For a perfectly calibrated model where `q = p`:

```
Perplexity = exp(H(p)) = exp(entropy of the data)
```

This is the **irreducible perplexity** — no model can do better than this.

### Typical Perplexity Values

| Model | Dataset | Perplexity |
|---|---|---|
| GPT-2 Small | WikiText-103 | ~29.4 |
| GPT-2 XL | WikiText-103 | ~17.5 |
| GPT-3 (175B) | WikiText-103 | ~16.4 |
| LLaMA 2 (7B) | WikiText-103 | ~11.3 |
| Human (estimated) | WikiText-103 | ~12-20 |

### Why Human Perplexity Isn't Zero

Language is inherently uncertain. Even a perfect language model can't predict the next word with certainty — there are multiple valid continuations. The entropy of natural language sets a floor on perplexity.

### Perplexity Pitfalls

- **Dataset dependent**: perplexity on different datasets is not comparable.
- **Tokenization dependent**: different tokenizers produce different perplexities for the same model.
- **Not a quality metric**: low perplexity ≠ good instruction following or reasoning.
- **Overfitting**: a model can have low perplexity on test data but still hallucinate.

### Bits Per Token

An alternative to perplexity that's more interpretable:

```
Bits per token = Cross Entropy / ln(2)
               = log₂(perplexity)
```

- GPT-2 on WikiText-103: ~4.9 bits/token
- Each token needs ~5 bits to encode — that's ~32 equally likely choices per token.

---

## LayerNorm Gradient Analysis

### Forward

```
y = γ * (x - μ) / √(σ² + ε) + β
```

### Backward (simplified)

```
∂L/∂x = (γ / √(σ² + ε)) * (∂L/∂y - mean(∂L/∂y) - (x - μ) * mean(∂L/∂y * (x - μ)) / (σ² + ε))
```

Key observations:
- The gradient is **modulated** by `γ / √(σ² + ε)` — the learned scale.
- The gradient is **centered** (subtracts mean) and **normalized** — this is what stabilizes training.
- If `γ` is very small, gradients vanish. If `γ` is very large, gradients explode.

### Why This Stabilizes Training

Without normalization, activations can grow or shrink exponentially through layers. LayerNorm ensures that every layer receives inputs with consistent scale, preventing the exponential growth/decay that causes vanishing/exploding gradients.

---

## RMSNorm vs LayerNorm: Empirical Results

### Quality

| | LayerNorm | RMSNorm |
|---|---|---|
| Language modeling | Baseline | ~0.5-1% worse (negligible) |
| Training stability | Good | Good (similar) |
| Convergence speed | Baseline | Similar or slightly faster |

### Speed

| | LayerNorm | RMSNorm |
|---|---|---|
| FLOPs per norm | 2 passes (mean + var) | 1 pass (RMS) |
| Memory | Same | Same |
| Wall-clock speedup | Baseline | ~7-10% faster |

### Why RMSNorm Works Without Mean Centering

- In practice, activations in transformers are approximately zero-mean already (due to residual connections and initialization).
- The mean centering in LayerNorm is often redundant.
- Removing it saves compute without meaningful quality loss.
- The scale `γ` is the critical part — it lets the model control activation magnitude.

---

## Pre-Norm + RMSNorm (Modern LLM Standard)

The standard architecture in modern LLMs (LLaMA, Mistral, Gemma):

```python
def transformer_block(x):
    # Pre-norm + RMSNorm
    x = x + attention(rms_norm(x))
    x = x + ffn(rms_norm(x))
    return x
```

### Why This Combination

1. **Pre-norm**: clean residual path for gradient flow.
2. **RMSNorm**: cheaper than LayerNorm, similar quality.
3. **No β parameter**: fewer parameters, empirically not needed.
4. **Together**: enables stable training of 70B+ parameter models.

---

## L5 Interview Q&A

### Q: "Why not use BatchNorm in transformers?"

1. **Batch=1 at inference**: BatchNorm needs batch statistics, which don't exist at batch=1.
2. **Variable sequence lengths**: padding corrupts batch statistics.
3. **Autoregressive generation**: each step has different effective batch sizes.
4. **Training-inference gap**: BatchNorm uses running averages at inference, creating a train/test mismatch.

### Q: "What's the difference between perplexity and cross entropy?"

Perplexity = exp(cross entropy). Cross entropy is in nats (natural log), perplexity is in "effective vocabulary size". A perplexity of 30 means the model is as uncertain as if choosing uniformly among 30 words.

### Q: "Can perplexity be used to compare models with different tokenizers?"

**No.** Different tokenizers produce different token sequences for the same text, so perplexity is not directly comparable. Always compare models with the same tokenizer, or use bits-per-character (BPC) which is tokenizer-independent.

### Q: "How does RMSNorm save compute compared to LayerNorm?"

LayerNorm computes: `mean(x)`, `x - mean`, `var(x)`, `(x - mean) / sqrt(var)`, `γ * ... + β` — 2 passes over the data.

RMSNorm computes: `mean(x²)`, `x / sqrt(mean(x²))`, `γ * ...` — 1 pass over the data.

The savings are ~50% of normalization FLOPs. Since normalization is ~5-10% of transformer compute, the overall speedup is ~5%.

### Q: "What happens if you remove normalization entirely from a transformer?"

Training becomes unstable — activations grow exponentially through layers, gradients explode or vanish. For a 2-layer model, it might work. For a 32-layer model, training will not converge. Normalization is essential for deep transformers.

### Q: "How does γ (scale) behave during training?"

γ is typically initialized to 1 and learned during training. Over time, different layers develop different γ values — some layers learn to amplify their outputs, others to dampen them. This gives the model flexibility in controlling information flow through the network.

---

## Interview Sound Bites

- Perplexity = exp(cross entropy) = effective number of choices per token. Lower is better.
- Bits per token = log₂(perplexity) — more interpretable than raw perplexity.
- Perplexity is tokenizer-dependent and dataset-dependent — not comparable across different setups.
- LayerNorm normalizes across features (D dimension); BatchNorm normalizes across batch — that's why BatchNorm fails for LLMs.
- RMSNorm removes mean-centering from LayerNorm — ~50% fewer FLOPs, negligible quality loss.
- Modern LLMs use pre-norm + RMSNorm: clean residual path + cheap normalization = stable training at scale.
- Normalization is essential for deep transformers — without it, activations explode/vanish exponentially.
- γ (scale) is the critical learned parameter; β (shift) is often unnecessary (RMSNorm drops it).
