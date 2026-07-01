# Calculus for ML

## Why Calculus?

Machine learning is optimization. Optimization requires gradients. Gradients come from derivatives. Calculus is the foundation.

---

## Derivatives

### Definition

The derivative measures how a function changes as its input changes:

```
f'(x) = lim(h→0) [f(x+h) - f(x)] / h
```

### Key Rules

| Rule | Formula |
|---|---|
| Power rule | `d/dx[x^n] = n·x^(n-1)` |
| Sum rule | `d/dx[f+g] = f' + g'` |
| Product rule | `d/dx[f·g] = f'·g + f·g'` |
| Chain rule | `d/dx[f(g(x))] = f'(g(x))·g'(x)` |
| Constant | `d/dx[c] = 0` |

---

## Partial Derivatives

When a function has multiple inputs, the partial derivative measures change with respect to one variable while holding others constant:

```
∂f/∂x_i = rate of change of f w.r.t. x_i
```

The **gradient** is the vector of all partial derivatives:

```
∇f = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]
```

---

## Chain Rule (The Most Important Rule in ML)

### Why it matters

Neural networks are compositions of functions. Backpropagation is just the chain rule applied repeatedly.

### Intuition

```
x → f → g → h → loss
```

To know how `x` affects `loss`, multiply the local derivatives:

```
∂loss/∂x = ∂loss/∂h · ∂h/∂g · ∂g/∂f · ∂f/∂x
```

### Example: Linear Regression

```
y_pred = X @ w
loss = (y_pred - y)²
```

Chain rule:

```
∂loss/∂w = ∂loss/∂y_pred · ∂y_pred/∂w
         = 2(y_pred - y) · X
```

---

## Gradient Descent

### Core Idea

Move in the direction that **decreases** the loss:

```
w = w - lr · ∇L
```

- `lr` = learning rate (step size)
- `∇L` = gradient of loss w.r.t. weights

### Learning Rate Intuition

- Too large → overshoot, diverge
- Too small → painfully slow convergence
- Just right → smooth descent

---

## Optimization Algorithms

See `optimization_algorithms.md` for the full evolution: Batch GD → SGD → Mini-Batch → Momentum → Adam.
