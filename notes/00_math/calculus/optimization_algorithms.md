# Optimization Algorithms: From Gradient Descent to Adam

## The Big Picture

Optimization algorithms answer one question: **Once we've computed the gradients using backpropagation, how should we update the model weights?**

The evolution of optimizers is a story where each new optimizer fixes a limitation of the previous one:

```
Batch Gradient Descent
  │  Too slow
  ▼
Stochastic Gradient Descent (SGD)
  │  Too noisy
  ▼
Mini-Batch Gradient Descent
  │  Oscillates
  ▼
Momentum
  │  One global learning rate
  ▼
Adam
```

---

## 1. Batch Gradient Descent

### Idea

Use the **entire training dataset** to compute the gradient.

```
Entire Dataset → Compute Loss → Compute Gradient → Update Weights
```

Only **one update per epoch**.

### Update Rule

```python
gradient = compute_gradient(all_training_data)
w = w - lr * gradient
```

### Advantages
- Very accurate gradient
- Stable updates
- Smooth convergence

### Problems
- For 1,000,000 samples, the optimizer waits for all of them before making a single update
- For modern LLMs trained on trillions of tokens, this is impractical

---

## 2. Stochastic Gradient Descent (SGD)

### Idea

Use **one training sample** at a time.

```
Sample 1 → Update
Sample 2 → Update
Sample 3 → Update
```

### Update Rule

```python
for sample in dataset:
    gradient = compute_gradient(sample)
    w = w - lr * gradient
```

### Advantages
- Updates happen immediately
- Learns much faster initially
- No need to wait for the entire dataset

### Problems
- Every training sample produces a different estimate of the true gradient
- The optimizer keeps changing direction → training becomes noisy

---

## 3. Mini-Batch Gradient Descent

This is what **almost every modern deep learning system** uses.

Instead of 1 sample or 1 million samples, use 32/64/128/256/512 samples.

```
64 Samples → Average Loss → Average Gradient → Update
```

### Why does Mini-Batching work?

Different samples produce different gradients. When averaged, random noise cancels out. The shared signal remains. The resulting gradient is much closer to the true gradient.

### GPU Advantage

GPUs process many samples simultaneously. Sending 1 sample to an RTX A6000 wastes most of the GPU. Sending 64 or 256 samples keeps thousands of CUDA cores busy.

Mini-batches therefore improve both:
- Computational efficiency
- Gradient quality

### Comparing the Three

| Optimizer | Batch Size | Updates per Epoch | Gradient Quality |
|---|---|---|---|
| Batch GD | Entire Dataset | 1 | Excellent |
| SGD | 1 | One per sample | Very Noisy |
| Mini-Batch GD | 32–1024 | Many | Good |

---

## Two Types of Oscillation

### 1. Data Oscillation

Occurs because different training samples produce different gradients.

```
Cat → Gradient →
Dog → Gradient ←
Car → Gradient ↗
```

Mini-batch gradient descent reduces this by averaging multiple samples.

### 2. Optimization Oscillation

Even with the perfect gradient, the loss landscape itself may cause oscillation.

Imagine a narrow valley:

```
Very Steep      ╲
                 ╲
                  X
                 ╱
Very Steep      ╱
```

Gradient Descent repeatedly jumps from one side of the valley to the other. Progress toward the minimum becomes slow.

**Momentum** was invented to reduce this type of oscillation.

---

## 4. Momentum

### Problem

Gradient Descent only looks at **today's gradient**. It completely forgets previous updates. This causes zig-zagging in narrow valleys.

### Idea

Remember the recent history of gradients. Introduce a variable **velocity** (`m`) which stores a moving average of past gradients.

### Update

```python
m = beta1 * m + (1 - beta1) * gradient
w = w - lr * m
```

### Intuition

- Gradient Descent asks: *"Where should I move today?"*
- Momentum asks: *"Where have I been moving recently?"*

Consistent directions are reinforced. Rapid left-right oscillations are dampened.

---

## 5. Adam

Momentum solves oscillation, but still uses **one global learning rate** for every parameter.

### Problem

Suppose:
- Weight A gradient = 100
- Weight B gradient = 0.001

Using the same learning rate causes huge updates for Weight A and tiny updates for Weight B. One learning rate does not fit every parameter.

### Key Insight

Momentum tells us **where** to move. We also need to know **how large** the gradients usually are.

Instead of storing average gradient, store **average squared gradient**. This is called the **Second Moment**.

### Why square the gradients?

Suppose gradients are `100, -100, 100, -100`. The mean is `0`, which incorrectly suggests small gradients. Squaring gives `10000, 10000, 10000, 10000` — now we correctly know the gradients are consistently large.

### Adam stores two statistics

**First Moment** (average direction):
```python
m = beta1 * m + (1 - beta1) * gradient
```

**Second Moment** (average gradient magnitude):
```python
v = beta2 * v + (1 - beta2) * gradient**2
```

**Weight Update:**
```python
w = w - lr * m / (sqrt(v) + eps)
```

### Interpretation

- `m` decides the **direction**
- `√v` scales the **step size**
- `eps` prevents division by zero
- Large gradients → Large `v` → Smaller update
- Small gradients → Small `v` → Larger effective update

---

## Summary

| Optimizer | Solves |
|---|---|
| Batch GD | Accurate gradients |
| SGD | Faster updates |
| Mini-Batch GD | Balances speed and stability |
| Momentum | Reduces optimization oscillation using gradient history |
| Adam | Adds adaptive learning rates for every parameter |

---

## Interview Takeaways

**Why Mini-Batch instead of SGD?**
- Reduces noisy gradients
- Better GPU utilization
- Faster convergence

**Why Momentum?**
- Reduces oscillation in narrow valleys
- Uses past gradients to smooth updates

**Why Adam?**
- Combines Momentum with adaptive learning rates
- Uses the first moment for direction
- Uses the second moment for scaling the update
- Handles parameters with different gradient magnitudes much better than SGD or Momentum
