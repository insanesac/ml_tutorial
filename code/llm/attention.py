import numpy as np

    
def mask_score(scores):
    N = scores.shape[0]
    
    mask = np.triu(np.ones((N, N)), k = 1)
    scores[mask == 1] = -1e9
    return scores
    
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    return exp/exp_sum

def scaled_dot_product_attention(Q, K, V):
    if Q.shape[1] != K.shape[1]:
        raise ValueError(...)
    
    scores = np.dot(Q, K.T) / (np.sqrt(K.shape[1]))
    scores = mask_score(scores.copy())  # mask attention
    weights = softmax(scores)
    outputs = weights @ V
    
    return outputs


def cross_entropy(y_true, y_pred):
    eps = 1e-9
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -np.sum(y_true * np.log(y_pred))

def layer_norm(x):
    mean = np.mean(x)
    variance = np.mean((x-mean)**2)

    eps = 1e-15
    std = np.sqrt(variance+eps)
    
    
    return (x-mean)/std

def rms_norm(x):
    eps = 1e-15
    rms = np.sqrt(np.mean(np.pow(x,2))+eps)
    
    return x/rms