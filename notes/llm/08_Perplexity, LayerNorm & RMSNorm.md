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
