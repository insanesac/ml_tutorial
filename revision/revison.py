import numpy as np

def softmax(X):
    X -= np.max(X, axis=-1, keepdims=True)
    exp_x = np.exp(X)
    
    return exp_x/(np.sum(exp_x, axis=-1, keepdims=True))

def cross_entropy(y_true, y_pred):
    # formula = - sum(y_true*log(y_pred)
    
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1-eps)
    return -1 * np.sum(y_true*np.log(y_pred))

def perplexity(ce):
    return np.exp(ce)

def layer_norm(x, gamma, beta, eps = 1e-15):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(variance+eps)
    

    x = (x -mean)/(std)
    
    return gamma*x + beta

def rms_norm(x, gemma):
    mean = np.mean(x**2,axis=-1,keepdims=True)
    mse = np.sqrt(mean+1e-5)
    
    x = x/mse
    
    return gemma*x

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b,axis=-1))

def retreival(query, doc, k):
    query = query/(np.linalg.norm(query)+1e-10)
    doc = doc/(np.linalg.norm(doc,axis=-1,keepdims=True)+1e-10)
    
    similarity = doc@query
    
    top_k_index = np.argpartition(similarity,-k)[-k:]
    top_k_index = top_k_index[np.argsort(similarity[top_k_index])][::-1]
    
    return top_k_index
    
def temperature_sampling(logits, T):
    return softmax(logits/T)

def top_k_sampling(
    probs,
    k
):
    top_k_index = np.argpartition(probs,-k)[-k:]
    top_k_index = top_k_index[np.argsort(probs[top_k_index])][::-1]
    
    top_k_probs = probs[top_k_index]/np.sum(probs[top_k_index])
    
    return np.random.choice(top_k_index, p=top_k_probs)
    
    