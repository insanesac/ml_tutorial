import numpy as np

def softmax(x):
    # for numerical stability else softmax will saturate
    x = x - np.max(x, axis=-1,keepdims=True)
    
    exp = np.exp(x)
    
    return exp/np.sum(exp,axis=-1,keepdims=True)

def cross_entropy(y_pred, y_true):
    eps = 1e-5
    
    y_pred =  np.clip(y_pred,eps, 1-eps)
    
    return -1 * np.sum(y_true * np.log(y_pred),axis=-1)

def perplexity(c_e):
    return np.exp(c_e)

def layer_norm(x, gamma, beta):
    eps = 1e-5
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.pow(x-mean,2),axis=-1, keepdims=True)
    std = np.sqrt(variance + eps) 
    
    x = (x-mean)/std
    
    return gamma*x + beta

def rms_norm(x, gamma):
    eps = 1e-5
    mse = np.sqrt(np.mean(x**2,axis=-1,keepdims=True)) 
    
    x = x/(mse+eps)
    
    return gamma*x
    
def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a,axis=-1) * np.linalg.norm(b,axis=-1))

def retreival(query, document, k):
    query = query/np.linalg.norm(query,axis=-1)
    document = document/np.linalg.norm(document,axis=-1,keepdims=True)
    
    similarity = document @ query
    
    top_k_index = np.argpartition(similarity, -k)[-k:]
    top_k_index = top_k_index[np.argsort(similarity[top_k_index])][::-1]
    
    return top_k_index

def temperature(logits, T):
    return softmax(logits/T)

def top_k_sampling(prob, k):
      
    top_k_index = np.argpartition(prob, -k)[-k:]
    
    top_k_probs = prob[top_k_index]/np.sum(probs[top_k_index])

    
    return np.random.choice(top_k_probs,p=prob)
    
def top_p_sampling(prob, p):
    sorted_indices = np.argsort(prob)[::-1]
    sorted_probs = prob[sorted_indices]
    
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum,p)+1
    
    top_indices = sorted_indices[:cutoff]
    top_p_probs = prob[top_indices]
    top_p_probs = top_p_probs / np.sum(top_p_probs)
    
    return np.random.choice(top_indices, p=top_p_probs)

