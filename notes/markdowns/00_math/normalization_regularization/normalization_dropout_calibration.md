# Deep Learning Foundations II: BatchNorm, Dropout & Calibration

## The Big Picture

After solving optimization and activation functions, researchers focused on three new problems:

```
Unstable Training       → Batch Normalization
Overfitting             → Dropout
Overconfident Predictions → Calibration
```

Each technique solves a completely different problem.

---

## 1. Batch Normalization

### Motivation

During training, every layer keeps changing its weights. As a result, the distribution of activations seen by the next layer constantly changes.

```
Epoch 1:   [-1.2,  0.8,  1.5, -0.6]
Epoch 20:  [  53,   61,   48,   57]
Epoch 40:  [-120,  -95, -103, -110]
```

Every layer must continuously adapt to changing input distributions. This is **Internal Covariate Shift**.

### Solution

Normalize every feature across the mini-batch:

```
x → Normalize → x̂ → γx̂ + β

where:
  γ (gamma) = learnable scaling
  β (beta)  = learnable shift
```

### Matrix View

```
          F1   F2   F3
S1         2    5    7
S2         3    8    6
S3         1    4    5
S4         6    7    9
```

BatchNorm normalizes each **column** (feature) using statistics computed from all samples in the batch.

### Advantages

- Faster convergence
- Stable gradients
- Allows larger learning rates
- Mild regularization effect

### Problem

BatchNorm depends on multiple samples. During autoregressive inference (batch size = 1), each feature column contains only one value — no meaningful batch statistics can be estimated.

---

## 2. LayerNorm (Comparison)

Instead of normalizing columns, LayerNorm normalizes each **row** independently.

```
→→→→  2   5  -1   7   (normalize this token's features)
→→→→  4   8   0   9   (normalize this token's features)
```

Each token is normalized independently using its own feature statistics.

### BatchNorm vs LayerNorm

| | BatchNorm | LayerNorm |
|---|---|---|
| Normalizes | Columns (features) | Rows (samples) |
| Statistics | Batch statistics | Feature statistics |
| Batch size | Depends on batch size | Independent of batch size |
| Best for | CNNs | Transformers |

### RMSNorm

**Observation:** Do we really need to subtract the mean? Often no.

Instead, normalize only by the root mean square.

### Advantages

- Simpler
- Faster
- Used in Llama and other modern LLMs

### Evolution

```
No Normalization → BatchNorm → LayerNorm → RMSNorm
```

---

## 3. Dropout

### Motivation

Neural networks often become overly dependent on a few neurons (**co-adaptation**).

```
A \      B      / C
    \    /    /
     \  /  /
      \/  /
       B
```

If neuron B learns everything, A and C become lazy. If B fails, the network performs poorly.

### Solution

During training, randomly disable neurons.

```
Before:              [2, 5, 1, 8]
After Dropout (50%): [2, 0, 1, 0]
```

The remaining neurons are forced to learn useful representations. No neuron can assume another neuron will always be available.

### Important

Dropout is used **only during training**. During inference, all neurons are active.

### Why Scaling is Needed

```
Training:   ~50% of neurons survive
Inference:  100% of neurons active
```

Without correction, activation magnitudes become inconsistent.

### Inverted Dropout

Modern implementations scale surviving activations during training:

```
keep_probability = 0.5
output = output / 0.5
```

This keeps the expected activation approximately the same during training and inference.

### Advantages

- Prevents overfitting
- Reduces co-adaptation
- Improves generalization
- Acts as regularization

---

## 4. Confidence Calibration

### Motivation

A classifier predicts "Cat 99%". But over thousands of examples, predictions with 99% confidence are actually correct only 82% of the time. The model is **overconfident**.

### Well-Calibrated Model

**Confidence should match reality:**

```
Confidence ≈ Accuracy
```

If the model predicts 90% confidence, it should be correct roughly 90% of the time.

---

## 5. Temperature Scaling

### Question

Can we reduce overconfidence **without retraining**?

### Solution

Apply temperature before softmax:

```
Softmax(logits / T)
```

- Higher temperature → softer probabilities → lower confidence
- Prediction (argmax) usually remains unchanged

### Two Uses of Temperature

| | LLM Sampling | Calibration |
|---|---|---|
| Controls | Randomness | Confidence |
| Higher T | More diverse outputs | Softer probabilities |
| Same equation | Yes | Yes |
| Purpose | Creativity | Trustworthiness |

---

## 6. Label Smoothing

### Motivation

Cross entropy trains with `[1, 0]`. The model learns to predict `0.999999` for every training sample. This produces overconfident models.

### Solution

Instead of `[1, 0]`, train using `[0.9, 0.1]` or `[0.95, 0.05]`.

The model learns: "This is almost certainly the correct class, but don't become infinitely confident."

### Advantages

- Better calibration
- Less overconfidence
- Improved generalization
- Regularization effect

### Label Smoothing vs Temperature Scaling

| | Label Smoothing | Temperature Scaling |
|---|---|---|
| When | During training | After training |
| Modifies | Labels | Logits |
| Effect | Prevents overconfidence | Corrects overconfidence |
| Retraining | Required | Not needed |

---

## Summary

| Technique | Problem | Solution |
|---|---|---|
| BatchNorm | Changing activation distributions | Normalize each feature across the batch |
| LayerNorm | BatchNorm fails for Transformers / batch=1 | Normalize each sample independently |
| RMSNorm | LayerNorm can be simplified | Normalize by root mean square only |
| Dropout | Overfitting and co-adaptation | Randomly disable neurons during training |
| Calibration | Overconfident predictions | Measure whether confidence matches accuracy |
| Temperature Scaling | Incorrect confidence after training | Scale logits before softmax |
| Label Smoothing | Cross entropy encourages extreme probabilities | Train using softened target labels |

---

## Evolution

```
Activation Functions
  → Need stable optimization → BatchNorm
  → Need Transformer-friendly norm → LayerNorm
  → Need simpler norm → RMSNorm
  → Need regularization → Dropout
  → Need trustworthy probabilities → Calibration
  → Temperature Scaling
  → Label Smoothing
```
