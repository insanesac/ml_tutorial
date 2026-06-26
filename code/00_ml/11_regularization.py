"""Regularization: L1 (Lasso) and L2 (Ridge).

Regularization adds a penalty term to the loss function to prevent overfitting:
    Loss = ||Xw - y||^2 + lambda * penalty(w)

L2 (Ridge): penalty = ||w||^2 = sum(w_i^2)
- Shrinks all weights proportionally (dense, small weights).
- Closed-form solution: w = (X^T X + lambda * I)^-1 X^T y
- Always has a closed-form solution.

L1 (Lasso): penalty = ||w||_1 = sum(|w_i|)
- Pushes small weights to exactly zero (sparse solution).
- No closed-form solution (requires iterative solvers like coordinate descent).
- Useful for feature selection.

Why large weights are dangerous:
- Large weights amplify small input changes → unstable predictions.
- Regularization keeps weights small → smoother, more generalizable model.

Dataset: y = 2x + 1 + noise
"""

import numpy as np

# --- Dataset ---
X = np.array([1, 2, 3, 4, 5])
y = np.array([3.1, 4.9, 7.2, 8.8, 11.1])  # y ≈ 2x + 1 with small noise

# Design matrix: [x, 1] for y = w*x + b
X_design = np.column_stack([X, np.ones_like(X)])  # shape (5, 2)

# --- No regularization (ordinary least squares) ---
w_no_reg = np.linalg.lstsq(X_design, y, rcond=None)[0]
print(f"No regularization: w={w_no_reg[0]:.4f}, b={w_no_reg[1]:.4f}")

# --- L2 (Ridge) regularization ---
# Closed form: w = (X^T X + lambda * I)^-1 X^T y
# The lambda * I term shrinks weights toward zero.
lambda_l2 = 1.0
I = np.eye(X_design.shape[1])  # identity matrix (2, 2)
w_l2 = np.linalg.inv(X_design.T @ X_design + lambda_l2 * I) @ X_design.T @ y
print(f"L2 (lambda={lambda_l2}): w={w_l2[0]:.4f}, b={w_l2[1]:.4f}")

# --- L1 (Lasso) intuition ---
# L1 has no closed-form solution (requires iterative optimization).
# Key property: tends to produce exact zeros → feature selection.
print(f"\nL1 pushes small weights to exactly 0 (sparsity)")
print(f"L2 shrinks all weights proportionally (dense, small weights)")

# --- Why large weights are dangerous ---
# A model with large weights produces wildly unstable predictions.
X_test = np.array([6, 7, 8])
w_large = np.array([100, -50])  # artificially large weights
y_test_large = w_large[0] * X_test + w_large[1]
print(f"\nLarge weight example: predictions explode: {y_test_large}")
