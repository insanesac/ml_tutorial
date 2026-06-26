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

## QLoRA (Quantized LoRA)

### The Idea

QLoRA combines quantization + LoRA to enable fine-tuning of **very large models on a single GPU**:

```
1. Quantize base model to 4-bit (NF4) → fits in much less memory
2. Add LoRA adapters in FP16 → trainable parameters
3. Forward pass: dequantize 4-bit weights → compute → LoRA adjustment
4. Backward pass: gradients flow only through LoRA adapters
```

### NF4 (NormalFloat 4-bit)

QLoRA introduces NF4 — a 4-bit quantization format optimized for normally-distributed weights:

```
Standard INT4: uniform quantization [-8, -7, ..., 7] (equal spacing)
NF4: non-uniform quantization (denser near 0, where most weights are)
```

NF4 has 16 levels placed at the quantiles of a normal distribution, matching the actual weight distribution.

### Double Quantization

QLoRA also uses **double quantization** — quantizing the quantization constants themselves:

```
1. Quantize weights to 4-bit (need scale factors per group)
2. Quantize the scale factors to 8-bit (they're usually FP32)
3. Saves ~0.5 bits per parameter on average
```

### Memory Math

```
65B model:
  FP16:     130 GB (needs multiple A100s)
  INT4:     ~33 GB (fits on a single A100 80GB)
  QLoRA:    ~33 GB (base) + ~100 MB (LoRA adapters) = ~33 GB
```

QLoRA enabled fine-tuning a 65B model on a single 48GB GPU — previously impossible.

### Paged Optimizers

QLoRA uses **paged optimizers** — optimizer states are moved to CPU when GPU memory is full, and brought back when needed. This prevents OOM during training.

---

## LoRA Rank Selection

### What Rank to Choose?

| Rank (r) | Trainable Params | Use Case |
|---|---|---|
| 4-8 | Very few | Light task adaptation, style transfer |
| 16-32 | Moderate | General fine-tuning, instruction following |
| 64-128 | Many | Domain adaptation, complex reasoning |
| 256+ | Heavy | Approaching full fine-tuning quality |

### Rules of Thumb

- **Start with r=16** — good default for most tasks.
- **Increase r** if the task requires significant domain shift.
- **Decrease r** if the task is close to the base model's capabilities.
- **Alpha (scaling)**: typically set to `α = 2r` (so `α/r = 2`).
- **Apply to all linear layers** (not just attention) for best quality — Q, K, V, O, gate, up, down.

### Which Layers to Apply LoRA To?

| Configuration | Layers | Quality | Memory |
|---|---|---|---|
| Attention only | Q, V | Moderate | Lowest |
| All attention | Q, K, V, O | Good | Low |
| All linear | Q, K, V, O, gate, up, down | Best | Moderate |

Modern best practice: apply LoRA to **all linear layers** — the quality improvement is worth the slightly higher memory.

---

## Quantization Methods Deep Dive

### GPTQ (Post-Training Quantization)

```
1. Process weights column by column
2. For each column, find optimal quantization that minimizes output error
3. Update remaining columns to compensate for quantization error
4. Result: INT4 weights with minimal quality loss
```

- **One-shot**: no training needed, just calibration data.
- **Good for**: GPU inference (INT4 weights, FP16 activations).
- **Quality**: ~1-2% degradation at 4-bit.
- **Speed**: ~3x faster inference (memory bandwidth reduction).

### AWQ (Activation-Aware Weight Quantization)

```
1. Observe activation magnitudes on calibration data
2. Identify "salient" weight channels (high activation → important)
3. Scale up salient channels before quantization (protect important weights)
4. Scale down after quantization
```

- **Activation-aware**: protects weights that matter most.
- **Better than GPTQ** for some models.
- **Good for**: GPU inference, edge deployment.
- **Quality**: ~0.5-1% degradation at 4-bit.

### SmoothQuant

```
1. Observe activation ranges
2. "Smooth" the difficulty: move quantization challenge from activations to weights
3. Scale = sqrt(max(|activation|) / max(|weight|))
4. Both weights and activations quantized to INT8
```

- **Weight + activation quantization** (not just weights).
- **Good for**: INT8 inference with minimal quality loss.
- **Used in**: TensorRT-LLM, some serving frameworks.

### GGUF (llama.cpp format)

```
1. Quantize weights to various precisions (Q4_0, Q4_K, Q5_K, Q8_0)
2. Optimized for CPU/Apple Silicon inference
3. Single file format (model + tokenizer + metadata)
```

- **Good for**: CPU inference, Mac/Metal, edge deployment.
- **Quality**: varies by quant level (Q4_K_M is the sweet spot).
- **Not for GPU**: GPTQ/AWQ are better for GPU.

### Comparison

| Method | Bits | Target | Quality Loss | Best For |
|---|---|---|---|---|
| GPTQ | 4 | GPU | ~1-2% | GPU serving |
| AWQ | 4 | GPU | ~0.5-1% | GPU serving, edge |
| SmoothQuant | 8 | GPU | ~0.5% | INT8 serving |
| GGUF | 4-8 | CPU | ~1-3% | CPU, Mac, edge |
| NF4 (QLoRA) | 4 | Training | ~1% | Fine-tuning |

---

## Quantization Formats Reference

| Format | Bytes/param | 7B Model | 70B Model |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16/BF16 | 2 | 14 GB | 140 GB |
| INT8 | 1 | 7 GB | 70 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB |

### BF16 vs FP16

| | FP16 | BF16 |
|---|---|---|
| Range | ±65,504 | ±3.4×10³⁸ |
| Precision | 10 bits mantissa | 7 bits mantissa |
| Dynamic range | Moderate | Same as FP32 |
| Training | Needs loss scaling | No loss scaling needed |
| Used in | Older GPUs (V100) | Modern GPUs (A100, H100) |

**BF16 is preferred for training** — same dynamic range as FP32, no loss scaling needed. FP16 has more precision but smaller range, causing overflow/underflow during training.

---

## L5 Discussion: Fitting a 70B Model on Limited Hardware

Three-pronged approach:

1. **Quantization** — INT8/INT4 reduces 70B from ~280GB to ~70GB / ~35GB.
2. **LoRA** — avoids full weight copies during fine-tuning.
3. **KV Cache optimization** — controls memory growth during inference.

---

## L5 Interview Q&A

### Q: "How would you fine-tune a 70B model on a single 8-GPU node (8x80GB A100)?"

1. **QLoRA**: quantize base model to 4-bit NF4 (~35 GB).
2. **Distribute with FSDP**: shard the quantized model across 8 GPUs (~4.4 GB each).
3. **LoRA adapters in FP16**: ~200 MB per GPU.
4. **Gradient checkpointing**: trade compute for memory — recompute activations in backward.
5. **Paged optimizers**: spill optimizer state to CPU if needed.
6. **Total memory per GPU**: ~4.4 GB (weights) + ~2 GB (adapters+optimizer) + ~4 GB (activations) = ~10 GB — easily fits.

### Q: "When would you use LoRA vs full fine-tuning vs continued pretraining?"

| Method | When | Compute | Quality |
|---|---|---|---|
| LoRA | Task adaptation, style, format | Low (0.1-1% params) | Good |
| Full FT | Domain shift, new capabilities | High (100% params) | Better |
| Continued PT | New language, new domain | Very high (needs lots of data) | Best for domain |

**Decision tree:**
1. Is the task close to base model's capabilities? → LoRA
2. Does the model need to learn new domain knowledge? → Full FT or continued PT
3. Is there a large domain shift (e.g., English → medical)? → Continued PT
4. Limited compute? → LoRA or QLoRA

### Q: "What's the quality difference between LoRA and full fine-tuning?"

- **Small tasks** (style, format, instruction following): LoRA ≈ full FT (negligible difference).
- **Medium tasks** (domain adaptation, new skills): LoRA is 1-3% worse.
- **Large tasks** (new language, significant capability gain): LoRA is 5-10% worse — full FT needed.
- **Applying LoRA to all linear layers** with r=64 closes most of the gap.

### Q: "How does INT4 quantization affect inference speed?"

The speedup comes from **memory bandwidth**, not compute:
- INT4 weights are 4x smaller → 4x less data to load from HBM.
- During decode (memory-bound), this gives ~3x speedup.
- During prefill (compute-bound), the speedup is smaller (~1.5x).
- The weights are dequantized to FP16 on-the-fly in SRAM — no quality loss in compute.

### Q: "Can you merge LoRA adapters back into the base model?"

Yes — for inference, merge `W + (α/r) * B @ A` into a single weight matrix:

```python
def merge_lora(base_weight, lora_A, lora_B, alpha, r):
    return base_weight + (alpha / r) * (lora_B @ lora_A)
```

After merging, there's **zero inference overhead** — the merged weight is the same shape as the original. This is how LoRA is used in production: train with adapters, merge for deployment.

But: you can't un-merge easily. If you need to swap adapters at runtime, keep them separate (with a small overhead).

---

## Interview Sound Bites

- LoRA freezes the base model and trains low-rank adapters (A and B) — typically 0.1-1% of parameters.
- **QLoRA** = 4-bit quantized base + LoRA adapters → fine-tune 70B on a single GPU.
- **NF4** (NormalFloat 4-bit): non-uniform quantization matched to weight distribution.
- **Rank selection**: r=16 default, r=64 for complex tasks, apply to all linear layers for best quality.
- **GPTQ**: post-training, column-wise, GPU inference. **AWQ**: activation-aware, protects salient channels.
- **SmoothQuant**: INT8 weight+activation quantization for serving. **GGUF**: CPU/edge inference.
- **BF16 > FP16 for training** — same dynamic range as FP32, no loss scaling.
- Memory math: 7B in INT4 = 3.5 GB, 70B in INT4 = 35 GB.
- LoRA adapters can be **merged** into base weights for zero-overhead inference.
- LoRA for task adaptation, full FT for domain shift, continued PT for new domains.
