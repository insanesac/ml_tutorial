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

## Key Gradient Result

Softmax + CE simplifies to:

`dL/dz = p - y`

This is why almost every framework exposes a single fused loss API.

## Interview Sound Bites

- Always use logits-based CE in production/training.
- It is mathematically equivalent to softmax + NLL but numerically stable.
- The fused backward pass gives clean gradient: `p - y`.
