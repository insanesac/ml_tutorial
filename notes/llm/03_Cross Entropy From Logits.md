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

## Key Gradient Result

Softmax + CE simplifies to:

`dL/dz = p - y`

This is why almost every framework exposes a single fused loss API.

## Interview Sound Bites

- Always use logits-based CE in production/training.
- It is mathematically equivalent to softmax + NLL but numerically stable.
- The fused backward pass gives clean gradient: `p - y`.
