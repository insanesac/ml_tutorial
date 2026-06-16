# LoRA & Quantization

## LoRA (Low-Rank Adaptation)

### Motivation

Full fine-tuning updates every weight in the model.

For a 7B parameter model that is expensive in memory and compute.

LoRA freezes the original weights and learns a **low-rank update** instead.

### Core Idea

Weight update decomposed as:

```
ΔW = A @ B

A.shape = (D, r)
B.shape = (r, D)
ΔW.shape = (D, D)
```

Applied as:

```
W_new = W + ΔW
```

During training: `W` is frozen. Only `A` and `B` are updated.

### Parameter Reduction

| Method | Parameters |
|---|---|
| Full Fine-Tuning | D² |
| LoRA | 2 * D * r |

**Example with D=4096, r=8:**

| Method | Params |
|---|---|
| Full | ~16.7M |
| LoRA | ~65K |

That is ~256x fewer trainable parameters.

### Where It Is Applied

Typically injected into attention weight matrices:

- `W_q` (query projection)
- `W_k` (key projection)
- `W_v` (value projection)
- `W_o` (output projection)

### Implementation Sketch

```python
class LoRALayer:
    def __init__(self, W, r):
        D = W.shape[0]
        self.W = W              # frozen
        self.A = np.random.randn(D, r) * 0.01
        self.B = np.zeros((r, D))

    def forward(self, x):
        return x @ (self.W + self.A @ self.B).T
```

### Interview Sound Bites

- Freeze base model, train low-rank adapters only.
- Parameter reduction: O(D²) → O(D * r).
- Merge `A @ B` into `W` at inference time for zero overhead.
- `r` is a hyperparameter; typical values: 4, 8, 16.

---

## Quantization

### Motivation

Reduce memory footprint and inference cost by storing weights at lower precision.

### Precision Formats

| Format | Bits | Bytes per param |
|---|---|---|
| FP32 | 32 | 4 |
| FP16 | 16 | 2 |
| BF16 | 16 | 2 |
| INT8 | 8 | 1 |
| INT4 | 4 | 0.5 |

### Memory Impact on a 7B Model

| Precision | Memory |
|---|---|
| FP32 | ~28 GB |
| FP16 | ~14 GB |
| INT8 | ~7 GB |
| INT4 | ~3.5 GB |

### Core Idea

```python
weights = [-1.2, 0.5, 1.7]

# Quantize
scale = 0.1
quantized = [round(w / scale) for w in weights]
# [-12, 5, 17]

# Dequantize
dequantized = [q * scale for q in quantized]
# [-1.2, 0.5, 1.7]
```

### Training vs Inference

| Phase | Typical Precision |
|---|---|
| Training | FP16 / BF16 |
| Inference | INT8 / INT4 |

### Tradeoffs

| Benefit | Risk |
|---|---|
| Less memory | Lower numerical precision |
| Faster inference | Potential quality degradation |
| Smaller deployment | Requires calibration data (PTQ) |

### Interview Sound Bites

- Quantization trades precision for memory and speed.
- INT8 and INT4 are the common inference formats.
- Excessive quantization degrades quality, especially on reasoning tasks.
- BF16 preserves FP32 dynamic range with 16-bit cost — preferred for training.

---

## L5 Discussion: Fitting a 70B Model on Limited Hardware

Three-pronged approach:

1. **Quantization** — INT8/INT4 reduces 70B from ~280GB to ~70GB / ~35GB.
2. **LoRA** — avoids full weight copies during fine-tuning.
3. **KV Cache optimization** — controls memory growth during inference.
