import numpy as np

# Bias-Variance dataset: y = x^2
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])

# Underfitting (high bias): linear model on quadratic data
def linear_fit(x, y):
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

# Good fit (low bias, low variance): quadratic model
X_poly = np.column_stack([X ** 2, X, np.ones_like(X)])
w_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]
y_pred_poly = X_poly @ w_poly
mse_poly = np.mean((y - y_pred_poly) ** 2)
print(f"\nQuadratic model: y = {w_poly[0]:.2f}x^2 + {w_poly[1]:.2f}x + {w_poly[2]:.2f}")
print(f"Quadratic MSE:   {mse_poly:.4f}  (good fit)")

# Overfitting (high variance): high-degree polynomial
X_over = np.column_stack([X ** i for i in range(6)])
w_over = np.linalg.lstsq(X_over, y, rcond=None)[0]
y_pred_over = X_over @ w_over
mse_over = np.mean((y - y_pred_over) ** 2)
print(f"\nDegree-5 model MSE: {mse_over:.4f}  (zero training error, high variance)")
