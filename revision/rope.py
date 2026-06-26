from numpy import matrix
import torch

def rope(X, cos, sin):
    even = X[...,::2]
    odd = X[...,1::2]
    
    pair1 = cos * even - sin * odd
    pair2 = cos * odd +  sin * even
    
    output = torch.empty_like(X)
    
    output[...,::2] = pair1
    output[...,1::2] = pair2

    return output

def build_rope(seq_len, head_dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    positions = torch.arange(seq_len).float()
    
    angles = torch.outer(positions, inv_freq)
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

def apply_rope(seq_len, head_dim, q, k):
    cos, sin = build_rope(seq_len, head_dim)
    
    q_rotated = rope(q, cos, sin)
    
    k_rotated = rope(k, cos, sin)
    
    return q_rotated, k_rotated



D must be even cause we are creating pairs. and each pair then has a shape of D/2
attention scores are calculated using q nad k. adding positonal information to v does not affect the attention score 
rope works cause we are rotating q nad k to embed positional information! [cos -sin, sin cos] is a rotation matrix
B N T D/2 or a broadcastable D/2
cos and sin needs to be calculated just once since its not dependant on the input X