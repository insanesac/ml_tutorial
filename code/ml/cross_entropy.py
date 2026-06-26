import numpy as np

# Softmax probabilities for 3 classes, 6 samples
# Small values -> class 0, medium -> class 1, large -> class 2

logits = np.array([
    [2.0, 0.5, 0.1],
    [2.2, 0.4, 0.2],
    [0.2, 2.0, 0.5],
    [0.3, 2.2, 0.4],
    [0.1, 0.5, 2.0],
    [0.2, 0.4, 2.2]
])

y_true = np.array([0, 0, 1, 1, 2, 2])

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

probs = softmax(logits)
m = len(y_true)

loss = -np.mean(np.log(probs[range(m), y_true]))
print(f"Cross-entropy loss: {loss:.4f}")

# Confidently wrong case
logits_wrong = np.array([
    [0.1, 0.5, 2.0],  # model thinks class 2, true is 0
    [2.0, 0.5, 0.1],  # model thinks class 0, true is 1
])
y_wrong = np.array([0, 1])

probs_wrong = softmax(logits_wrong)
loss_wrong = -np.mean(np.log(probs_wrong[range(len(y_wrong)), y_wrong]))
print(f"Confidently wrong loss: {loss_wrong:.4f}")
