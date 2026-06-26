# Cross Entropy From Logits

## Why From Logits (Not Probabilities)?

In practice, we do **not** do:
1. `p = softmax(logits)`
2. `loss = -log(p[y])`

directly, because it can be numerically unstable.

Instead, we compute cross-entropy from logits using the log-sum-exp trick.

## Stable Formula

For one example with class `y`:

`CE = -z_y + log(sum_j exp(z_j))`

where `z` is the logits vector.

## Why This Is Better

- Avoids overflow in `exp(large)`
- Avoids underflow in tiny probabilities
- Avoids `log(0)` issues

## Log-Sum-Exp Trick

`log(sum(exp(z))) = m + log(sum(exp(z - m)))` where `m = max(z)`

Subtracting max keeps exponentials bounded.

## NumPy Implementation (Single Example)

```python
def ce_from_logits(logits, y_true_idx):
    m = np.max(logits)
    logsumexp = m + np.log(np.sum(np.exp(logits - m)))
    return -logits[y_true_idx] + logsumexp
```

## Batch Version (One-Hot Labels)

```python
def ce_from_logits_batch(logits, y_onehot):
    # logits: (B, C), y_onehot: (B, C)
    m = np.max(logits, axis=1, keepdims=True)
    logsumexp = m + np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True))
    log_probs = logits - logsumexp
    ce = -np.sum(y_onehot * log_probs, axis=1)
    return np.mean(ce)
```

## Step-by-Step (Sparse Labels, Interview Style)

### What Are We Given?

Not probabilities. We are given logits and class indices.

```python
logits = np.array([
    [2, 1, 0],
    [5, 3, 1]
])
# shape: (B, C)

labels = np.array([1, 0])
# shape: (B,)
```

### Naive Approach

```python
probs = softmax(logits)
loss = -np.log(probs[np.arange(B), labels])
```

This works, but can overflow for large logits (e.g., `exp(1000)`).

### Stable Approach

Instead of `softmax -> log -> cross entropy`, compute CE directly from logits.

### Step 1: Extract correct-class logits

```python
B = logits.shape[0]
z_correct = logits[np.arange(B), labels]
```

For the example above, `z_correct = [1, 5]`.

### Step 2: Compute stable logsumexp

```python
m = np.max(logits, axis=-1, keepdims=True)
shifted = logits - m

logsumexp = (
    np.log(np.sum(np.exp(shifted), axis=-1))
    + m.squeeze(-1)
)
```

### Step 3: Per-sample loss

```python
loss = -z_correct + logsumexp
```

### Step 4: Batch mean

```python
final_loss = np.mean(loss)
```

### Full Sparse Implementation

```python
def ce_from_logits_sparse(logits, labels):
    # logits: (B, C), labels: (B,)
    B = logits.shape[0]

    z_correct = logits[np.arange(B), labels]

    m = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    logsumexp = np.log(np.sum(np.exp(shifted), axis=-1)) + m.squeeze(-1)

    loss = -z_correct + logsumexp
    return np.mean(loss)
```

## PyTorch Fused Implementation

In practice, never compute softmax + log + NLL separately. Use the fused API:

```python
import torch
import torch.nn.functional as F

# F.log_softmax + F.nll_loss = F.cross_entropy
loss = F.cross_entropy(logits, labels)  # labels: (B,) integer indices

# This is equivalent but less efficient:
log_probs = F.log_softmax(logits, dim=-1)
loss = F.nll_loss(log_probs, labels)
```

### Why Fused Is Better

1. **Numerical stability**: PyTorch's `F.cross_entropy` uses the log-sum-exp trick internally.
2. **Speed**: one kernel launch instead of three (softmax, log, NLL).
3. **Memory**: no intermediate probability matrices materialized.
4. **Gradient**: fused backward computes `p - y` directly without materializing the Jacobian.

---

## Gradient Computation from Logits

### Forward

```
L = -z_y + log(Σ_j exp(z_j))
```

### Backward (per-sample)

```
∂L/∂z_k = -δ_{ky} + exp(z_k) / Σ_j exp(z_j)
         = -δ_{ky} + p_k
         = p_k - y_k
```

Where `δ_{ky}` is 1 if `k == y` else 0, and `p_k = softmax(z_k)`.

### Batch Gradient

```
∂L/∂Z = P - Y    # shape: (B, C)

# P = softmax(Z, dim=-1)
# Y = one_hot(labels)
```

Then `mean` reduction divides by batch size:

```
∂L_mean/∂Z = (P - Y) / B
```

---

## Vocabulary-Level Cross Entropy (LLM Specific)

In LLMs, the "classes" are vocabulary tokens. For a sequence of `N` tokens:

```
Logits shape: (B, N, V)    # V = vocab size (e.g., 128,000)
Labels shape: (B, N)       # integer token IDs
```

### Shifted Labels (Causal LM)

The model predicts token `t+1` from tokens `0..t`. So labels are shifted:

```python
# logits: (B, N, V), labels: (B, N)
shift_logits = logits[:, :-1, :]    # predictions for positions 0..N-2
shift_labels = labels[:, 1:]        # targets are positions 1..N-1

loss = F.cross_entropy(
    shift_logits.reshape(-1, V),    # (B*(N-1), V)
    shift_labels.reshape(-1)         # (B*(N-1),)
)
```

### Why Shifting Matters

- Position 0 predicts token at position 1
- Position 1 predicts token at position 2
- ...
- Position N-2 predicts token at position N-1
- Position N-1 has no target (or is padding)

Forgetting to shift is one of the most common LLM training bugs.

---

## Memory Considerations for Large Vocabularies

For V=128,000 (LLaMA 3):

```
Logits for one token: 128,000 * 4 bytes (FP32) = 512 KB
Logits for batch=8, seq=4096: 8 * 4096 * 128,000 * 4 = 16 GB (!!!)
```

### Mitigations

- **Use FP16/BF16**: halves the memory
- **Compute CE in chunks**: process tokens in groups
- **Fused kernels**: never materialize full softmax probabilities
- **Vocabulary parallelism**: split the output projection across GPUs (tensor parallelism)

---

## L5 Interview Q&A

### Q: "Write cross entropy from logits in PyTorch without using F.cross_entropy"

```python
def ce_from_logits(logits, labels):
    # logits: (B, C), labels: (B,)
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    loss = -log_probs[torch.arange(logits.shape[0]), labels]
    return loss.mean()
```

`torch.logsumexp` is the stable way to compute `log(Σ exp(x))`.

### Q: "What's the memory bottleneck during LLM training with CE loss?"

The logits tensor. For a batch of `B` sequences of length `N` with vocab `V`:

```
Logits memory = B * N * V * bytes_per_param
```

For B=8, N=4096, V=128K, FP16: `8 * 4096 * 128000 * 2 = 8 GB` — just for logits, per transformer layer's output projection.

This is why **tensor parallelism** splits the vocabulary dimension across GPUs.

### Q: "Why is F.cross_entropy faster than manual softmax + log + NLL?"

1. **Kernel fusion**: one CUDA kernel instead of three — less kernel launch overhead.
2. **No intermediate tensors**: softmax probabilities (B, C) are never materialized in HBM.
3. **Optimized backward**: gradient `p - y` computed directly from logits.
4. **Memory bandwidth**: fewer HBM reads/writes = faster on memory-bound workloads.

### Q: "How would you implement CE for a 128K vocabulary efficiently?"

1. Use fused `F.cross_entropy` — never materialize softmax.
2. If memory is still an issue, chunk the sequence dimension.
3. For distributed training, use vocabulary parallelism (split logits across GPUs, all-reduce the logsumexp).
4. Consider FP16/BF16 logits to halve memory.

### Q: "What goes wrong if you forget to shift labels in causal LM training?"

The model learns to predict the **current** token from the **current** context (including itself). This is trivial — the model just copies the input embedding through to the output. It learns nothing useful about language modeling and the loss collapses to near-zero for the wrong reason.

---

## Key Gradient Result

Softmax + CE simplifies to:

`dL/dz = p - y`

This is why almost every framework exposes a single fused loss API.

---

## Interview Sound Bites

- Always use logits-based CE in production/training — never softmax + log separately.
- It is mathematically equivalent to softmax + NLL but numerically stable via log-sum-exp.
- The fused backward pass gives clean gradient: `p - y` — no Jacobian materialization.
- For LLMs, don't forget to shift labels: predict token t+1 from tokens 0..t.
- The logits tensor is the memory bottleneck for large vocabularies — use fused kernels and tensor parallelism.
- `torch.logsumexp` is the stable primitive — know it cold.
