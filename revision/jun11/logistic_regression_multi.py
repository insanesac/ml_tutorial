from this import d
import numpy as np

# Multi-Feature Logistic Regression
# Both features increase with class label
# Expected: w1 > 0, w2 > 0, b < 0

X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [3, 3],
    [4, 3],
    [4, 4]
])

y = np.array([0, 0, 0, 1, 1, 1])

w = np.zeros(X.shape[1])
b = 0

lr = 0.01
epochs = 10000

eps = 1e-15

for _ in range(epochs):
    z = X@w + b
    
    p = 1 / (1 + np.exp(-z))
    p = np.clip(p, eps, 1 - eps)
    
    L = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
    print(L)
    
    dw = (X.T @ (p - y)) / len(X)
    db = np.mean(p - y)
    
    w -= lr * dw
    b -= lr * db
    
print(w,b)