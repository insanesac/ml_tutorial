import torch
import math
import torch.nn as nn

import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.q_proj = nn.Linear(d_model, d_model)

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.head_dim = d_model // self.num_heads

        self.group_size = num_heads // num_kv_heads 

        self.k_proj = nn.Linear(d_model, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_kv_heads)


        self.out_proj = nn.Linear(d_model, d_model)


    def causal_mask(self, x, T):
        mask = torch.triu(torch.full((T,T),float("-inf"), device=x.device), diagonal=1)
        
        return mask 

    def attention(self, Q, K, V, mask=None):
        # F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores += mask

        A = F.softmax(scores, dim=-1)

        return torch.matmul(A, V)


    def forward(self, x):
        B, T, D = x.shape

        mask = self.causal_mask(x, T)

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        output = self.attention(Q, K, V, mask)

        output = output.transpose(1, 2).reshape(B, T, D)

        return self.out_proj(output)

