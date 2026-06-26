import torch
import torch.nn as nn
import math

class MQA(nn.Module):
    def __init__(self,d_model, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)

        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads

        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)

        
    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        
        scores = torch.matmul(Q, K.tranpose(-2,-1))

        scores = scores / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
            
        A = torch.softmax(scores, dim=-1)
        
        return torch.matmul(A, V)
        
        
    def forward(self, x):
        B, N, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(B, N, self.num_heads, self.head_dim). transpose(2,1)
        K = K.view(B, N, 1, self.head_dim). transpose(2,1)
        V = V.view(B, N, 1, self.head_dim). transpose(2,1)
            
            
        K = K.expand(-1, self.num_heads, -1, -1)
        V = V.expand(-1, self.num_heads, -1, -1)
        
        output = self.attention(Q, K, V)
        
        output = output.transpose(2,1).reshape(B, N, D)
        
        return self.out_proj(output)
    
    
    
B T D
B H T Dh for Q and B 1 T Dh for K and V
we stack k and v so that it matches the num_heads
instead of calculting K and V for every query, we calculate it once.iter
it does not
tradeoff is accuracy drop