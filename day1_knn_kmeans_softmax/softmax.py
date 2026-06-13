import numpy as np

X = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10],
    [20, 20],
    [21, 20]
])

y = np.array([0, 0, 1, 1, 2, 2])

W = np.random.randn(X.shape[1], 3) * 0.01
b = np.zeros(3)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

epochs = 1000
lr = 0.1

for i in range(epochs):
    z = X @ W + b
    p = softmax(z)

    m = X.shape[0]
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    dz = p.copy()
    dz[range(m), y] -= 1
    dz /= m

    dW = X.T @ dz
    db = np.sum(dz, axis=0)

    W -= lr * dW
    b -= lr * db

print(f"\nFinal predictions:\n{np.argmax(softmax(X @ W + b), axis=1)}")
