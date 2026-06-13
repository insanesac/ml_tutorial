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

# Forward pass
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = softmax(z2)

print(f"Output shape: {a2.shape}")
print(f"Predictions: {np.argmax(a2, axis=1)}")
print(f"True labels: {y}")
