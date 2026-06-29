# Efficient LLMs: Training & Inference Optimizations

## Training vs Inference Optimizations

| Category | Techniques | Goal |
|---|---|---|
| **Training** | Mixed Precision, Gradient Checkpointing, Prompt Tuning, Prefix Tuning, LoRA, QLoRA | Reduce GPU memory and training cost |
| **Inference** | Continuous Batching, Streaming Generation, KV Cache, Quantization | Improve latency, throughput, deployment efficiency |

---

## 1. Continuous Batching

### Motivation

Traditional batching performs poorly during autoregressive generation — different requests finish at different times:

```
Request A → 10 tokens
Request B → 500 tokens
Request C → 120 tokens

Static batching:
A B C → A finishes → B C → C finishes → B
GPU utilization steadily decreases.
```

### Solution

Completed requests are **immediately replaced** with new ones:

```
A B C → A finishes → B C D → C finishes → B D E → D finishes → B E F
GPU remains almost fully occupied.
```

### Why It Works

Each request maintains its own KV Cache. When a request completes:
1. Remove its KV Cache
2. Insert a new request
3. Continue decoding

No recomputation required.

### Benefits

- Higher GPU utilization
- Better throughput
- Lower latency under heavy load

**Does it improve model quality?** No — purely an inference optimization.

---

## 2. Streaming Generation

### Motivation

Without streaming, the user waits until the entire response is generated. Streaming sends tokens **immediately**:

```
Generate Token → Send Token → Generate Token → Send Token → ...
```

### Time To First Token (TTFT)

Users care much more about TTFT than Time To Last Token. Streaming dramatically reduces perceived latency.

### Benefits

- Better user experience
- Lower perceived latency
- Users can interrupt generation early

### Continuous Batching + Streaming

Modern inference servers combine both:

```
Incoming Requests
  ↓
Continuous Batch Scheduler
  ↓
GPU → One Decoding Step
  ↓
Immediately Stream Tokens
```

---

## 3. Mixed Precision

### Motivation

FP32 training consumes significant GPU memory. Many computations can safely use 16-bit floating point.

### FP32 vs FP16

| | FP32 | FP16 |
|---|---|---|
| Bits | 32 | 16 |
| Precision | More | Less |
| Memory | Higher | Lower |
| Speed | Slower | Faster |

### Why Not FP16 Everywhere?

Very small gradients may **underflow** to zero → training becomes unstable.

### Mixed Precision Solution

- Use **FP16/BF16** for forward pass and backward pass
- Keep **FP32 master weights** for optimizer updates

### Loss Scaling

Very small gradients are multiplied before backpropagation:

```
Loss: 0.0002 → Scaled: 0.2
Backpropagation uses larger values.
Gradients divided by same scale before updating weights.
```

### Automatic Mixed Precision (AMP)

PyTorch provides `torch.autocast()` and `GradScaler()` for automatic mixed precision.

### FP16 vs BF16

| | FP16 | BF16 |
|---|---|---|
| Exponent range | Smaller | Same as FP32 |
| Stability | Less | Better |
| Preferred for | Older GPUs | Modern LLMs |

### Benefits

- Reduced GPU memory
- Faster training
- Larger batch sizes
- Similar model accuracy

---

## 4. Gradient Checkpointing

### Motivation

Backpropagation requires intermediate activations. Standard training stores **every** activation:

```
Forward: Store A1, Store A2, Store A3, Store A4
```

Activation memory becomes enormous for deep Transformers.

### Solution

Store only **selected checkpoints**:

```
Store: A2, A5, A8
All other activations discarded.
```

During backpropagation, missing activations are **recomputed**:

```
Checkpoint → Forward Recompute → Missing Activation → Continue Backprop
```

### Trade-off

| | Standard | Gradient Checkpointing |
|---|---|---|
| Memory | High | Low |
| Computation | 1x forward | ~2x forward |
| Training speed | Faster | Slower |

### Mixed Precision + Gradient Checkpointing

```
Mixed Precision → Smaller tensors
Gradient Checkpointing → Fewer stored tensors
Together → Significantly reduced GPU memory
```

---

## 5. Prompt Tuning

### Motivation

Fine-tuning billions of parameters for every downstream task is expensive.

### Idea

Learn only a small set of trainable **prompt embeddings**:

```
User Prompt → Learned Prompt + User Prompt
```

The learned prompt consists of trainable embedding vectors, not actual words.

### Training

- Only prompt embeddings are updated
- The Transformer remains **completely frozen**

### Advantages

- Extremely small number of trainable parameters
- Easy deployment (swap prompts, not model weights)
- Very memory efficient

### Drawbacks

- Only influences the model through input embeddings
- Less expressive than Prefix Tuning or LoRA

---

## 6. Prefix Tuning

### Motivation

Prompt embeddings influence only the input. Their effect may diminish through many Transformer layers.

### Idea

Learn trainable **Key and Value prefixes** for **every Transformer layer**:

```
Normal Attention:    Q, K, V
Prefix Tuning:       Q, [Prefix K + K], [Prefix V + V]
```

The learned prefixes participate in attention computation throughout the network.

### Advantages

- More expressive than Prompt Tuning (influences every layer)
- Still keeps the Transformer frozen
- Very parameter efficient

### Drawbacks

- Slightly more trainable parameters than Prompt Tuning

---

## PEFT Comparison

| Method | Trainable Parameters |
|---|---|
| Full Fine Tuning | Entire Model |
| Prompt Tuning | Prompt Embeddings |
| Prefix Tuning | Prefix K/V Vectors |
| LoRA | Low-Rank Adapter Matrices |
| QLoRA | LoRA Adapters + Quantized Base Model |

## Memory Optimization Techniques

| Technique | Primary Benefit |
|---|---|
| Mixed Precision | Smaller tensors |
| Gradient Checkpointing | Fewer stored activations |
| LoRA | Fewer trainable weights |
| QLoRA | Quantized frozen weights |
| Smaller Batch Size | Fewer activations |
| Shorter Context Length | Smaller attention memory |
| Distributed Training | Model split across GPUs |

## Complete Interview Cheat Sheet

| Topic | Primary Goal | Training/Inference |
|---|---|---|
| Continuous Batching | GPU Utilization | Inference |
| Streaming Generation | Lower Perceived Latency | Inference |
| Mixed Precision | Reduce Memory | Training |
| Gradient Checkpointing | Reduce Activation Memory | Training |
| Prompt Tuning | Efficient Fine-Tuning | Training |
| Prefix Tuning | More Expressive PEFT | Training |
