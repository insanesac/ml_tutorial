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

## KL Divergence Connection

Cross entropy and KL divergence are closely related:

```
H(p, q) = H(p) + D_KL(p || q)
```

Where:
- `H(p, q)` = cross entropy of true distribution `p` and predicted `q`
- `H(p)` = entropy of true distribution (constant for fixed labels)
- `D_KL(p || q)` = KL divergence = how much `q` differs from `p`

**Since `H(p)` is constant during training, minimizing cross entropy = minimizing KL divergence.**

This means cross entropy is literally pushing the predicted distribution toward the true distribution.

### Why This Matters for Interviews

- CE is not arbitrary — it's the natural loss when you want to match a probability distribution.
- KL divergence is the "extra information" (in bits) needed to encode samples from `p` using a code optimized for `q`.
- Minimizing CE = minimizing surprise = minimizing information waste.

---

## Why Cross Entropy and Not MSE?

| | Cross Entropy | MSE |
|---|---|---|
| Gradient with softmax | `p - y` (clean, linear) | Complex, saturating gradient |
| Behavior on wrong predictions | Heavy penalty (log → -∞) | Quadratic penalty (bounded) |
| Optimization landscape | Smoother, better conditioned | Flat gradients near correct answer |
| Theoretical justification | Maximum likelihood | Least squares (Gaussian assumption) |

### The Key Problem with MSE + Softmax

When using MSE with softmax, the gradient becomes:

```
dL/dz = 2 * (p - y) * p * (1 - p)  (simplified)
```

The `p * (1 - p)` term means the gradient **vanishes** when the model is very confident (p → 0 or p → 1), even if it's confidently wrong. Cross entropy's gradient `p - y` has no such vanishing — it stays large until the prediction is correct.

---

## Information Theory Intuition

```
Entropy H(p)     = expected surprise = -Σ p_i * log(p_i)
Cross Entropy    = -Σ p_i * log(q_i) = surprise under wrong model
KL Divergence    = D_KL(p||q) = extra surprise from using q instead of p
```

- **Entropy**: average number of bits needed to encode the true distribution.
- **Cross entropy**: average number of bits needed if you use model `q`'s code to encode data from `p`.
- **Minimizing CE**: making your model's code as efficient as possible for the true data.

### Example

```
True distribution:     [1, 0, 0, 0]  (certain it's class 0)
Model prediction:      [0.7, 0.2, 0.1, 0]

CE = -1*log(0.7) - 0 - 0 - 0 = 0.357 bits of "wasted" capacity
```

If model predicts `[0.99, 0.01, 0, 0]`:

```
CE = -1*log(0.99) = 0.01 bits — nearly perfect
```

---

## Temperature in Cross Entropy

Temperature is applied to logits **before** softmax, not to probabilities:

```
p_i = softmax(z_i / T)
```

- **T < 1**: sharper distribution, model more confident
- **T > 1**: flatter distribution, model less confident
- **T → 0**: approaches one-hot (argmax)
- **T → ∞**: approaches uniform

### When Temperature Matters

- **Training**: temperature is usually 1.0 (no scaling) during standard training.
- **Distillation**: teacher uses high T to produce soft labels; student trains on soft targets.
- **Inference**: controls diversity vs determinism of generation.

---

## Label Smoothing Deep Dive

Instead of one-hot `[1, 0, 0, ...]`, use:

```
y_smooth = (1 - ε) * one_hot + ε / C
```

Where `ε` is typically 0.1 and `C` is the number of classes.

### Why It Works

- Prevents the model from becoming **overconfident** (logit → ∞).
- Acts as regularization — the model can never achieve zero loss, so it doesn't push logits to extremes.
- Improves calibration — predicted probabilities are more reliable.
- Used in: GPT-2, ViT, many state-of-the-art models.

### Without Label Smoothing

The model can drive `p_correct → 1` by pushing logits to infinity. This:
- Makes the model overconfident
- Degrades calibration
- Can hurt generalization

---

## Focal Loss (Alternative to CE for Imbalanced Data)

```
FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
```

Where:
- `p_t` = predicted probability of correct class
- `γ` = focusing parameter (typically 2)
- `α` = class weight

### Intuition

- `(1 - p_t)^γ` down-weights easy examples (high `p_t`)
- Model focuses on **hard** examples (low `p_t`)
- Used in object detection (RetinaNet) where background examples dominate

### When to Use vs CE

| | Cross Entropy | Focal Loss |
|---|---|---|
| Balanced data | ✅ | Overkill |
| Imbalanced data | OK with class weights | Better |
| Many easy examples | Wastes gradient on easy | Down-weights easy |
| Calibration | Good | Can hurt calibration |

---

## L5 Interview Q&A

### Q: "Derive why softmax + CE gives gradient p - y"

```
L = -Σ y_i * log(p_i)
p_i = softmax(z_i) = exp(z_i) / Σ exp(z_j)

dL/dz_k = Σ_i y_i * (p_k - δ_{ik})    [chain rule through softmax]
        = p_k * Σ y_i - y_k
        = p_k * 1 - y_k              [since Σ y_i = 1 for one-hot]
        = p_k - y_k
```

The key insight: the Jacobian of softmax is `diag(p) - p p^T`, and when multiplied by the one-hot label vector, most terms cancel.

### Q: "Why is cross entropy called 'cross' entropy?"

Because it's the entropy computed using the **cross** distribution — you're evaluating the code (log probabilities) from model `q` on data from distribution `p`. It's "cross" because the two distributions are different (ideally converging).

### Q: "What happens if you use CE without softmax?"

You get negative log-likelihood (NLL) directly:

```
L = -log(p_y)
```

This works if `p` is already a valid probability distribution (e.g., from sigmoid for binary classification). But for multi-class, softmax is needed to ensure probabilities sum to 1.

### Q: "How does label smoothing affect the gradient?"

With label smoothing, the target becomes `(1-ε) * one_hot + ε/C`. The gradient changes from `p - y` to `p - y_smooth`:

```
dL/dz = p - [(1-ε) * y + ε/C]
```

This means even when `p = y` (perfect prediction), the gradient is non-zero: `ε/C - ε * 1 = ε(1/C - 1)`. This prevents logits from growing unbounded.

### Q: "When would you use binary CE vs categorical CE?"

- **Binary CE**: 2 classes (or multi-label where each class is independent). Uses sigmoid.
- **Categorical CE**: mutually exclusive multi-class. Uses softmax.
- **Multi-label**: each class is independent (an image can have both "cat" and "outdoor" labels). Use sigmoid + BCE per class, not softmax + CE.

---

## 30-Second Explanation

Cross entropy measures model confidence in the correct class. Combined with softmax, it produces the elegant gradient `p - y`, is numerically stable with log-sum-exp tricks, and reduces to `-log(p_correct)` under one-hot labels. It's equivalent to minimizing KL divergence between predicted and true distributions, making it the natural loss for maximum likelihood estimation in classification.
