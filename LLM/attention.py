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