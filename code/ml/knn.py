import numpy as np

X_train = np.array([
    [1, 1],
    [2, 1],
    [10, 10],
    [11, 10]
])

y_train = np.array([0, 0, 1, 1])

x_new = np.array([1.5, 1.5])

k = 3

distances = np.linalg.norm(X_train - x_new, axis = 1)
    
dist_index = np.argsort(distances)

k_index = dist_index[:k]

neighbor_labels = y_train[k_index]

prediction = np.argmax(
    np.bincount(neighbor_labels)
)


print(prediction)