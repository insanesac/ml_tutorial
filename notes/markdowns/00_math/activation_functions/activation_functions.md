# Activation Functions Evolution

## The Big Picture

Every new activation function was invented to solve a limitation of the previous one:

```
Linear
  ↓  No non-linearity
Sigmoid
  ↓  Vanishing gradients, not zero-centered
Tanh
  ↓  Still saturates
ReLU
  ↓  Dying neurons
Leaky ReLU
  ↓  Hard threshold, not smooth
GELU
  ↓  Higher computational cost
SiLU (Swish)
```

---

## 1. Linear Activation

```
y = Wx + b
```

**Problem:** Stacking multiple linear layers produces another linear transformation.

```
Layer 1:  y = W₁x + b₁
Layer 2:  z = W₂y + b₂
Substitute:  z = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = Wx + b
```

100 linear layers = 1 linear layer. Deep networks gain no expressive power.

**Motivation:** Neural networks require non-linearity. Without it, deep learning does not exist.

---

## 2. Sigmoid

```
Range: 0 → 1
σ(x) = 1 / (1 + e^(-x))
```

### Advantages

- Smooth and differentiable
- Ideal for binary probability outputs

### Problems

- Not zero-centered
- Saturates for large positive/negative inputs
- Causes vanishing gradients

**Today mainly used for:** binary classification outputs, gates in LSTM/GRU.

---

## 3. Tanh

```
Range: -1 → 1
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

### Motivation

Improve upon Sigmoid by making activations zero-centered.

### Advantages

- Zero-centered outputs
- Better optimization than Sigmoid

### Problem

Still saturates for large magnitudes. Vanishing gradients remain.

---

## 4. ReLU

```
ReLU(x) = max(0, x)
```

### Advantages

- Extremely simple and fast
- No saturation for positive values
- Enabled successful training of deep networks

### Problem: Dying ReLU

For negative inputs:

```
Output = 0
Derivative = 0
Weight update: w = w − η∇L  where ∇L = 0
→ No learning occurs
→ Neuron may permanently output zero
```

**Motivation:** Can we prevent neurons from dying?

---

## 5. Leaky ReLU

```
Instead of:  negative → 0
Use:         negative → 0.01x
```

### Advantages

- Gradient never becomes exactly zero
- Neurons can recover
- Reduces dying ReLU problem

### Remaining Problem

Still has a hard threshold at zero. Not smooth.

**Motivation:** Can activation become smooth instead of making hard ON/OFF decisions?

---

## 6. GELU (Gaussian Error Linear Unit)

### Key Idea

Instead of asking "Is x positive?", ask "How likely should this neuron activate?"

Activation becomes **probabilistic** rather than binary.

### Characteristics

- Smooth
- Allows partial activation
- Small positive values activate weakly
- Small negative values are not immediately discarded

### Advantages

- Better optimization
- More expressive than ReLU
- Excellent for NLP

### Used by

BERT, GPT-2, GPT-3

### Limitation

More computationally expensive than ReLU.

---

## 7. SiLU (Swish)

```
SiLU(x) = x · sigmoid(x)
```

### Intuition

| Input | Behavior |
|---|---|
| Large positive | Almost unchanged |
| Large negative | Gradually suppressed |
| Small values | Partially activated |

### Advantages

- Smooth everywhere
- No dead neurons
- Excellent gradient flow
- Computationally efficient
- Empirically better for very deep transformers

### Used by

Llama, Mistral, Qwen — most modern open-source LLMs.

---

## Evolution Summary

| Activation | Innovation | Problem Solved | Remaining Problem |
|---|---|---|---|
| Linear | Basic mapping | — | No non-linearity |
| Sigmoid | Non-linearity, probability | Linear limitation | Vanishing gradients, not zero-centered |
| Tanh | Zero-centered | Sigmoid saturation | Still saturates |
| ReLU | Fast, no positive saturation | Vanishing gradients | Dead neurons |
| Leaky ReLU | Small negative gradients | Dead neurons | Hard threshold, not smooth |
| GELU | Smooth probabilistic gating | Hard threshold | Higher compute cost |
| SiLU | Smooth, efficient, stable | Compute cost | Current state-of-the-art |

---

## Interview Takeaways

| Function | One-liner |
|---|---|
| Sigmoid | Best for binary probabilities |
| Tanh | Zero-centered alternative to Sigmoid |
| ReLU | Made deep learning practical. Main weakness: dead neurons |
| Leaky ReLU | Allows small gradients for negative inputs |
| GELU | Smooth probabilistic gating. Preferred by BERT and GPT |
| SiLU | Smooth, efficient, stable. Current SOTA for most modern LLMs |

---

## One-Line Story

```
Linear → Need non-linearity
  → Sigmoid → Need zero-centered outputs
    → Tanh → Need faster deep training
      → ReLU → Need to avoid dead neurons
        → Leaky ReLU → Need smooth activations
          → GELU → Need smooth with lower compute cost
            → SiLU
```
