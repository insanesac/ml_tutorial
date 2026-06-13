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

