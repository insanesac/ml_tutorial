# Backpropagation Notes for LLM Interviews

## Goal

Backpropagation answers one question:

> How should each parameter change to reduce the loss?

It computes the gradient of the loss with respect to every trainable parameter.

---

## Intuition

**Forward pass:**

```
Input → Linear → Prediction → Loss
```

If the prediction is wrong, we need to determine:
- Which parameters caused the error?
- By how much should they change?
- In which direction should they change?

The gradient answers all three.

---

## Gradient vs Gradient Descent

- The **gradient** points in the direction of the steepest increase in loss (uphill).
- **Gradient descent** minimizes the loss by moving in the opposite direction.

**Weight update:**

```
W = W - learning_rate × gradient
```

If the gradient is positive:
```
Gradient = +5
W_new = W - lr × 5    →  Weight decreases
```

If the gradient is negative:
```
Gradient = -5
W_new = W - lr × (-5) = W + lr × 5    →  Weight increases
```

**Remember:** Gradient points uphill. Gradient descent walks downhill.

---

## Simple Linear Regression Example

### Prediction

```
ŷ = Wx + b
```

### Error

```
e = ŷ - y = Wx + b - y
```

### Loss (Mean Squared Error)

```
L = e²
```

### Chain Rule

Instead of differentiating everything at once:

```
Weight → Prediction → Error → Loss
```

We compute:

```
dL/dW = dL/de × de/dW
```

This is the **Chain Rule**.

### Gradient with Respect to W

```
L = (Wx + b - y)²

dL/dW = 2(Wx + b - y)x
```

or:

```
Gradient = 2 × Error × Input
```

**Key intuition:**
- Larger error → larger update
- Larger input → larger update
- Zero error → zero gradient
- Zero input → zero gradient

### Gradient with Respect to Bias

```
dL/db = 2(Wx + b - y)
```

or:

```
Gradient = 2 × Error
```

Bias is independent of the input.

---

## Computational Graph

Every neural network is simply a graph of operations.

```
Input → Linear → Activation → Loss
```

Backpropagation walks through this graph in reverse:

```
Loss ← Activation ← Linear ← Input
```

Every operation computes gradients for the operation before it.

---

## Backpropagation in an LLM

**Forward pass:**

```
Tokens → Embedding → Transformer Layers → Hidden States → LM Head (Linear) → Logits → Softmax → Cross Entropy → Loss
```

**Backward pass:**

```
Loss ← Cross Entropy ← Softmax ← LM Head ← Transformer ← Embeddings
```

Nothing special happens because it's an LLM. Backprop simply follows the computational graph backwards.

---

## Softmax + Cross Entropy

**Model output:**

```
Logits → Softmax → Probabilities → Cross Entropy → Loss
```

The resulting gradient simplifies to:

```
Gradient = p - y
```

where:
- `p` = predicted probabilities
- `y` = one-hot target

### Example

```
Prediction:  [0.63, 0.23, 0.14]
Target:      [0,    1,    0]
Gradient:    [0.63, -0.77, 0.14]
```

- Positive gradients reduce incorrect logits.
- Negative gradients increase the correct logit.

---

## Why Does PyTorch Store Activations?

During backpropagation, gradients depend on values computed during the forward pass.

### Linear Layer

```
y = Wx + b
Gradient: dL/dW = Error × Input
```

PyTorch must remember the input `x`. Without it, the gradient cannot be computed.

### ReLU

```
Forward:  y = max(0, x)
Backward: if x > 0 → gradient = 1, else → gradient = 0
```

PyTorch must remember whether `x` was positive.

### Softmax

Backward uses `p - y`. Therefore the probabilities from the forward pass must be available.

### LayerNorm

Forward computes:
- Mean
- Variance
- Normalized values

Backward requires these same values. Hence they are stored.

---

## Why Training Uses More Memory Than Inference

**Training:**

```
Forward → Store Activations → Backward → Delete Activations
```

**Inference:**

```
Forward → Output
```

No backward pass. No gradients. No computational graph. No activation storage.

Hence inference requires much less GPU memory.

---

## PyTorch Autograd

During the forward pass, PyTorch builds a computational graph automatically.

Calling `loss.backward()` causes PyTorch to traverse the graph backwards.

Each operation computes:
- Gradient with respect to its inputs
- Gradient with respect to its parameters

These gradients are accumulated into `parameter.grad`.

Finally, `optimizer.step()` updates every parameter:

```
parameter -= learning_rate * parameter.grad
```

---

## Interview Takeaways

- Gradient points uphill. Gradient descent moves downhill.
- Backpropagation is repeated application of the Chain Rule.
- Gradients depend on intermediate values from the forward pass.
- PyTorch stores activations because they are required during backpropagation.
- Training consumes more memory because activations and the computational graph must be retained until the backward pass.
- Every layer in a Transformer follows the same backpropagation principles. There is no special "Transformer backprop."
