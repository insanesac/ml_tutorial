import numpy as np
#Numerically Stable Softmax
X = np.array([2, 1, 0])

def softmax(x):
    x = x - np.max(x,axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis=1, keepdims=True)



def batched_softmax(logits):
    logits = np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis=1, keepdims=True)
