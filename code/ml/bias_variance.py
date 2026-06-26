"""Bias-Variance tradeoff demonstration.

Bias: error from incorrect assumptions (underfitting).
- High bias = model too simple to capture the pattern.
Variance: error from sensitivity to training data (overfitting).
- High variance = model memorizes noise, fails to generalize.

Tradeoff:
- Increase model complexity → bias ↓, variance ↑
- Decrease model complexity → bias ↑, variance ↓
- Goal: find the sweet spot (low bias + low variance).

Dataset: y = x^2 (quadratic relationship).
We fit 3 models of increasing complexity:
1. Linear (degree 1) — underfitting (high bias)
2. Quadratic (degree 2) — good fit (low bias, low variance)
3. Degree-5 polynomial — overfitting (high variance)
"""

import numpy as np

# --- Dataset: y = x^2 ---
X = np.array([1, 2, 3, 4, 5])    # shape (5,)
y = np.array([1, 4, 9, 16, 25])  # shape (5,)

# --- Underfitting: linear model on quadratic data ---
# A line cannot capture a curve → systematically wrong (high bias).
def linear_fit(x, y):
    """Fit y = w*x + b using closed-form OLS."""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - w * x_mean
    return w, b

w_lin, b_lin = linear_fit(X, y)
y_pred_lin = w_lin * X + b_lin
mse_lin = np.mean((y - y_pred_lin) ** 2)
print(f"Linear model: y = {w_lin:.2f}x + {b_lin:.2f}")
print(f"Linear MSE:   {mse_lin:.4f}  (high bias, underfitting)")

# --- Good fit: quadratic model ---
# Matches the true relationship → low bias, low variance.
# Design matrix: [x^2, x, 1] → solve via least squares.
X_poly = np.column_stack([X ** 2, X, np.ones_like(X)])  # shape (5, 3)
w_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]      # shape (3,)
y_pred_poly = X_poly @ w_poly
mse_poly = np.mean((y - y_pred_poly) ** 2)
print(f"\nQuadratic model: y = {w_poly[0]:.2f}x^2 + {w_poly[1]:.2f}x + {w_poly[2]:.2f}")
print(f"Quadratic MSE:   {mse_poly:.4f}  (good fit)")

# --- Overfitting: degree-5 polynomial ---
# With 5 data points and 6 parameters, model can fit perfectly.
# Zero training error but high variance — will fail on new data.
X_over = np.column_stack([X ** i for i in range(6)])  # shape (5, 6)
w_over = np.linalg.lstsq(X_over, y, rcond=None)[0]    # shape (6,)
y_pred_over = X_over @ w_over
mse_over = np.mean((y - y_pred_over) ** 2)
print(f"\nDegree-5 model MSE: {mse_over:.4f}  (zero training error, high variance)")
