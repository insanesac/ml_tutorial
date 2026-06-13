import numpy as np

# Dataset: y = 2x + 1 + noise
X = np.array([1, 2, 3, 4, 5])
y = np.array([3.1, 4.9, 7.2, 8.8, 11.1])

X_design = np.column_stack([X, np.ones_like(X)])

# No regularization
w_no_reg = np.linalg.lstsq(X_design, y, rcond=None)[0]
print(f"No regularization: w={w_no_reg[0]:.4f}, b={w_no_reg[1]:.4f}")

# L2 (Ridge) regularization: minimize ||Xw - y||^2 + lambda * ||w||^2
lambda_l2 = 1.0
I = np.eye(X_design.shape[1])
w_l2 = np.linalg.inv(X_design.T @ X_design + lambda_l2 * I) @ X_design.T @ y
print(f"L2 (lambda={lambda_l2}): w={w_l2[0]:.4f}, b={w_l2[1]:.4f}")

# L1 (Lasso) intuition: tends to push weights to exactly zero
# (Full L1 requires iterative solvers; shown here as conceptual)
print(f"\nL1 pushes small weights to exactly 0 (sparsity)")
print(f"L2 shrinks all weights proportionally (dense, small weights)")

# Why large weights are dangerous
X_test = np.array([6, 7, 8])
w_large = np.array([100, -50])
y_test_large = w_large[0] * X_test + w_large[1]
print(f"\nLarge weight example: predictions explode: {y_test_large}")
