import numpy as np

def softmax(X):
    eps = 1e-15
    X = np.clip(X, eps, 1- eps)
    exp = np.exp(X)
    
    return exp/(np.sum(eps,axis=1,))

