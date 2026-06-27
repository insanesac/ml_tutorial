# Linear Algebra

## Core Objects

| Object | Notation | Shape | Example |
|---|---|---|---|
| Scalar | `x` | `()` | `3.14` |
| Vector | `x` | `(D,)` | `[1.0, -2.3, 0.5]` |
| Matrix | `X` | `(N, D)` | batch of embeddings |
| Tensor | `X` | `(B, N, D)` | batch of sequences |

- **N** = number of samples
- **D** = number of features / embedding dimension
- **C** = number of classes

---

## Key Operations

### Matrix Multiplication

```
(A, B) @ (B, C) = (A, C)
```

Each element is a dot product of a row and a column. Used in every linear layer (`W @ x`), attention (`Q @ K.T`), and embedding lookup.

### Dot Product

```
a · b = Σ a_i * b_i
```

Measures similarity. Used in attention scores and cosine similarity.

### Matrix Transpose

```
X: (N, D)  →  X.T: (D, N)
```

Swaps rows and columns. Used in gradient computations (`X.T @ gradient`) and attention (`Q @ K.T`).

### Broadcasting

NumPy/PyTorch automatically expands smaller arrays to match larger ones:

```
(N, D) + (D,)    →  (D,) broadcast to (N, D)
(N, D) * (N, 1)  →  (N, 1) broadcast to (N, D)
```

**Rule:** Dimensions must either match or be 1. Broadcasting goes right-to-left.

---

## Shape Discipline

Always ask before writing code:

1. What is the input shape?
2. What is the output shape?
3. What intermediate shapes are needed?

This catches 90% of bugs in ML code.

---

## Complexity

| Operation | Time |
|---|---|
| Dot product | O(D) |
| Matrix-vector | O(ND) |
| Matrix-matrix | O(N·D·C) |
| Attention (Q @ K.T) | O(N²D) |

ML operations are often O(ND) because each sample has D features.
