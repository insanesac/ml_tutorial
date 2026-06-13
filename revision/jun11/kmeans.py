import numpy as np

X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [10, 10],
    [10, 11],
    [11, 10],
    [20, 20],
    [20, 21],
    [21, 20]
])

max_iter = 1000

k = 3

centroids = X[np.random.choice(X.shape[0], k, replace=False)].astype(float)

for _ in range(max_iter):
    old_centroids = centroids.copy()
    clusters = [[] for _ in range(k)]
    
    for x in X:
        distances = []
        for c in centroids:
            distances.append(np.linalg.norm(x-c))
            
        closest = np.argmin(distances)
        
        clusters[closest].append(x)
        
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            centroids[i] = np.mean(cluster, axis=0)
            
    if np.allclose(old_centroids, centroids):
        break
    
print(centroids)