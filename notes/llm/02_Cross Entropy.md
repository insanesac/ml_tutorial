# Cross Entropy

## What Problem Does It Solve?

Accuracy is blind to confidence.

A barely-correct prediction and a highly-confident correct prediction look identical under accuracy.

Cross entropy measures **how much probability mass the model assigns to the correct class**.

## Formula

**Multiclass:**

```
L = -Σ y_i * log(p_i)
```

**With one-hot labels** (only correct class survives):

```
L = -log(p_correct)
```

## Intuition

| Confidence on Correct Class | Loss |
|---|---|
| p = 1.0 | 0 |
| p = 0.9 | small |
| p = 0.5 | moderate |
| p = 0.1 | large |
| p → 0 | → +∞ |

Wrong and confident = huge penalty.

## Why Log?

- `log(a*b) = log(a) + log(b)` — products of probabilities become sums of log-probabilities.
- Prevents numerical underflow from chaining small probabilities.
- Heavily penalizes confident wrong predictions (steep curve near 0).

## Softmax

Converts raw logits into probabilities that sum to 1:

```
p_i = exp(z_i) / Σ exp(z_j)
```

## Numerical Stability

Two rules:

1. Clip probabilities before log:

```python
y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
```

2. Shift logits before softmax (softmax is shift-invariant):

```python
z = z - np.max(z)   # prevents overflow in exp
```

## Implementation

```python
def cross_entropy(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))
```

## Why One-Hot Works as a Mask

`y_true * log(y_pred)` zeroes out all incorrect class terms.

Only the correct class contributes to the loss.

## Softmax Derivatives

```
dp_i/dz_i = p_i * (1 - p_i)        # same index
dp_i/dz_j = -p_i * p_j             # different index
```

## Key Result

Softmax + Cross Entropy gradient simplifies to:

```
dL/dz = p - y
```

One of the most important results in ML. Know it cold.

## Complexity

- Time: `O(B * C)` where B = batch size, C = classes
- Extra space: `O(1)` for scalar loss

## Shape Reasoning

| Scenario | Logits | Gradient |
|---|---|---|
| Single example | `(C,)` | `(C,)` |
| Batch | `(B, C)` | `(B, C)` |

Reduced loss is always a scalar.

## Sparse Labels

When class count is large (e.g. 50k vocabulary):
- Store class indices instead of one-hot vectors.
- Use `CrossEntropyLoss(logits, labels)` directly.

## Edge Cases

- **Label smoothing** — prevents overconfidence, improves generalization.
- **Class imbalance** — model gravitates toward majority class. Fix with weighted losses.

## Common Traps

- Taking `log(0)` — always clip.
- Using raw logits when function expects probabilities.
- Dividing loss by N manually when framework already averages.

## 30-Second Explanation

Cross entropy measures model confidence in the correct class. Combined with softmax, it produces the elegant gradient `p - y`, is numerically stable with log-sum-exp tricks, and reduces to `-log(p_correct)` under one-hot labels.
