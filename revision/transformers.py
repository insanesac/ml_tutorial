import numpy as np

from LLM import attention

def rope(x, cos, sin):
    pair1 = x[...,::2]
    pair2 = x[...,1::2]
    
    even_rot = pair1 * cos - pair2 * sin
    odd_rot = pair1 * sin + pair2 * cos
    
    output = np.empty_like(x)
    
    output[...,::2] = even_rot
    output[...,1::2] = odd_rot
    
    return output

def update_cache(K_cache, new_K):
    return np.concatenate(
        [K_cache, new_K],
        axis=2
    )
    
def update_cache(cache, new):
    return np.concatenate(
        [cache, new],
        axis=2
    )
    
def multihead_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    B, N, D = X.shape

    assert D % num_heads == 0

    head_dim = D // num_heads

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv


    """
    only during inference
    K = np.empty_like(Q)
    V = np.empty_like(Q)
    K_new= X @ Wk
    V_new = X @ Wv

    K = update_cache(K, K_new)
    V = update_cache(V, V_new)
    """
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)  # B num N D
    K = K.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)

    output = attention(Q,K,V,mask)

    output = output.transpose(0,2,1,3).reshape(B,N,D)

    return output @ Wo


def mqa(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    '''
    Wq.shape = (D, D)

    Wk.shape = (D, head_dim)

    Wv.shape = (D, head_dim)
    '''
    B, N, D = X.shape

    assert D % num_heads == 0

    head_dim = D // num_heads

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)  # B num N D
    K = K.reshape(B, N, 1, head_dim).transpose(0,2,1,3)
    V = V.reshape(B, N, 1, head_dim).transpose(0,2,1,3)

    K = np.repeat(K, num_heads, axis=1)
    V = np.repeat(V, num_heads, axis=1)

    output = attention(Q,K,V,mask)
        
    output = output.transpose(0,2,1,3).reshape(B,N,D)

    return output @ Wo


def gqa(X, Wq, Wk, Wv, Wo, num_heads, group_size, mask=None):
    '''
    Wq.shape = (D, D)

    Wk.shape = (D, num_kv_heads * head_dim)

    Wv.shape = (D, num_kv_heads * head_dim)
    '''
    B, N, D = X.shape
    
    assert D % num_heads == 0

    head_dim = D // num_heads

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)  # B num N D
    K = K.reshape(B, N, num_heads//group_size, head_dim).transpose(0,2,1,3)
    V = V.reshape(B, N, num_heads//group_size, head_dim).transpose(0,2,1,3)
    
    K = np.repeat(K, group_size,axis=1)
    V = np.repeat(V, group_size,axis=1)
    
    output = attention(Q,K,V,mask)
    
    output = output.transpose(0,2,1,3).reshape(B,N,D)
    
    return output @ Wo


import numpy as np

def cross_entropy_from_logits(
    logits,
    labels
):
    """
    logits: (B, C)
    labels: (B,)
    """

    max_logits = np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    shifted = logits - max_logits

    logsumexp = (
        np.log(
            np.sum(
                np.exp(shifted),
                axis=-1
            )
        )
        + max_logits.squeeze(-1)
    )

    z_correct = logits[
        np.arange(logits.shape[0]),
        labels
    ]

    loss = -z_correct + logsumexp

    return np.mean(loss)

def lora_linear(x, W, A, B):
    return x @ W + x @ A @ B

def moe_forward(
    x,
    experts,
    W_router,
    k=2
):

    router_logits = x @ W_router

    router_probs = softmax(
        router_logits
    )

    topk_idx = np.argsort(
        router_probs
    )[::-1][:k]

    topk_probs = (
        router_probs[topk_idx]
        /
        np.sum(router_probs[topk_idx])
    )

    output = 0

    for p, idx in zip(
        topk_probs,
        topk_idx
    ):
        output += (
            p * experts[idx](x)
        )

    return output