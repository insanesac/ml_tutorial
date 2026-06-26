import numpy as np

z = np.array([-3, -1, 0, 1, 3, 5])

def relu(x):
    return np.maximum(0, x)

a = relu(z)
print(f"Input:  {z}")
print(f"ReLU:   {a}")

# Vanishing gradient intuition: sigmoid saturates, ReLU does not
z_large = np.array([10, 50, 100])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print(f"\nSigmoid at large values: {sigmoid(z_large)}")
print(f"ReLU at large values:    {relu(z_large)}")

# Dead ReLU problem
z_all_negative = np.array([-5, -3, -1])
print(f"\nReLU all negative: {relu(z_all_negative)} (all dead)")
