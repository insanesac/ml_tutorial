import torch
import torch.nn as nn
import math



class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        assert d_model % num_heads == 0
        
        self.head_dim = d_model // num_heads
        
        self.group_size = num_heads // num_kv_heads
        
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        
        
    def attention(self, Q, k, V, mask=None):
        dd_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2,-1))
        
        scores = scores / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
            
        A = torch.softmax(scores, dim=-1)
        
        return torch.matmul(A, V)
    
    def forward(self, x, mask=None):
        
        B, T, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)        

        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1,2)
        
                
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        output = self.attention(Q,K,V, mask)
        
        output = output.transpose(1,2).reshape(B, T, D)
        
        return self.out_proj(output)
    
    
