import numpy as np

# Multi-Feature Linear Regression
# True relationship: y = 2*x1 + 3*x2 + 1

X = np.array([
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 2],
    [5, 4]
])

y = np.array([9, 8, 16, 15, 23])

w = np.zeros(X.shape[1])
b = 0

epochs = 1000
lr = 0.01

for i in range(epochs):
    y_pred = X @ w + b
    error = y_pred - y

    loss = np.mean(error ** 2)
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    dw = X.T @ error / len(X)
    db = np.mean(error)

    w -= lr * dw
    b -= lr * db

print(f"\nFinal w: {w}")
print(f"Final b: {b:.4f}")
