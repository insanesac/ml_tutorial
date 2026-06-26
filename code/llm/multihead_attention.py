def mha(X, Wq, Wk, Wv, num_heads):

    B, N, D = X.shape

    assert D % num_heads == 0
    
    head_dim = D // num_heads
    
    
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    Q = Q.reshape(B,N,H,head_dim).transpose(0,2,1,3)

    K = K.reshape(B,N,H,head_dim).transpose(0,2,1,3)

    V = V.reshape(B,N,H,head_dim).transpose(0,2,1,3)

    scores = Q @ K.transpose(0,1,3,2)

    scores /= np.sqrt(head_dim)

    A = softmax(scores, axis = -1)

    out = A @ V

    out = out.transpose(0,2,1,3).reshape(B, N, H*head_dim)

    out = out @ Wo
    
    return out