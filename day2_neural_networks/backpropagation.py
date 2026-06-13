import numpy as np

# Reuse Softmax dataset
X = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10],
    [20, 20],
    [21, 20]
])

y = np.array([0, 0, 1, 1, 2, 2])

input_dim = 2
hidden_dim = 4
output_dim = 3

W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

epochs = 1000
lr = 0.1

for i in range(epochs):
    # Forward
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    # Loss
    m = X.shape[0]
    log_likelihood = -np.log(a2[range(m), y])
    loss = np.sum(log_likelihood) / m
    if i % 100 == 0:
        print(f"epoch {i}: loss = {loss:.4f}")

    # Backprop
    dz2 = a2.copy()
    dz2[range(m), y] -= 1
    dz2 /= m

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0).astype(float)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

print(f"\nFinal predictions: {np.argmax(softmax(relu(X @ W1 + b1) @ W2 + b2), axis=1)}")
