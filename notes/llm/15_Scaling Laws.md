# Scaling Laws

## What Are Scaling Laws?

Empirical relationships between model performance and three variables:

- **N** — number of parameters
- **D** — number of training tokens
- **C** — compute budget (FLOPs)

The key finding: loss follows predictable power laws as you scale any of these.

---

## OpenAI Scaling Laws (Kaplan et al., 2020)

Loss decreases as a power law with scale:

```
L(N) ∝ N^(-α)
L(D) ∝ D^(-β)
```

Key finding: **model size matters more than dataset size** for a fixed compute budget.

Recommendation: train large models on less data.

---

## Chinchilla Scaling Laws (Hoffmann et al., 2022)

Revisited Kaplan's findings with better experiments.

### Core Result

For a given compute budget C, the optimal strategy is:

```
N_optimal ∝ C^0.5
D_optimal ∝ C^0.5
```

**Parameters and tokens should scale equally.**

### Chinchilla Rule of Thumb

```
Optimal tokens ≈ 20 × N
```

To train a model with N parameters optimally, use ~20N training tokens.

### Examples

| Model | Params | Tokens Used | Chinchilla Optimal Tokens |
|---|---|---|---|
| GPT-3 | 175B | 300B | ~3.5T (undertrained) |
| Chinchilla | 70B | 1.4T | ~1.4T (optimal) |
| LLaMA 3 | 8B | 15T | (overtrained intentionally) |

### Why Overtrain Intentionally?

Chinchilla optimal = best loss for a given compute budget during training.

But for **inference**, a smaller model trained on more data is cheaper to serve.

LLaMA overtrains small models so they are inference-efficient.

---

## The Three Axes

| Axis | Effect | Bottleneck |
|---|---|---|
| More parameters (N) | Better capacity | Memory |
| More tokens (D) | Better generalization | Data collection |
| More compute (C) | Better overall | Cost |

---

## Emergent Abilities

At certain scale thresholds, capabilities appear suddenly:

- Few-shot learning
- Chain-of-thought reasoning
- Code generation

Not predictable from small-scale extrapolation.

Still not fully understood.

---

## Practical Implications for L5

| Question | Answer |
|---|---|
| Why use a 7B model over 70B for inference? | Cheaper, Chinchilla-overtrained small models can match larger undertrained ones |
| Why does GPT-3 underperform vs smaller newer models? | It was undertrained relative to its parameter count |
| How do you decide training token budget? | Chinchilla: ~20× parameter count |
| What limits scale in practice? | Compute cost, memory, data quality |

---

## Interview Sound Bites

- Scaling laws show that loss follows power laws with model size, data, and compute.
- Chinchilla showed parameters and tokens should scale equally for compute-optimal training.
- The Chinchilla rule: ~20 tokens per parameter for optimal training.
- Modern small models (LLaMA, Gemma) are intentionally overtrained for inference efficiency.
- Emergent abilities appear at scale thresholds and are not predictable from small runs.
