import math
import torch
import numpy as np
import torch.nn as nn
 
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, vocab_size, max_seq_len):
        super().__init__()
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        
        self.ffn1 = nn.Linear(d_model, 4*d_model)
        self.ffn2 = nn.Linear(4*d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        

        
    def softmax(self, X):
        X = X - torch.max(X, dim=-1, keepdim=True).values
        
        exp = torch.exp(X)
        
        return exp / torch.sum(exp, dim=-1, keepdim=True)
    
    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        
        scores = torch.matmul(Q, K.transpose(2,3))
        
        scores = scores / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
            
        A = self.softmax(scores)
        
        return torch.matmul(A, V)
    
    def mha(self, X, mask=None):
        B, N, D = X.shape

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)        
        
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1,2)      
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        
        output = self.attention(Q, K ,V, mask)
        
        output = output.transpose(1,2).reshape(B, N, D)
        
        return self.out_proj(output)
    
    def ffn_layer(self, X):
        hidden = torch.relu(self.ffn1(X))
        
        return self.ffn2(hidden)
    
    def transformer_block(self, X, mask=None):
        mha_out = self.mha(X, mask)
        
        X = self.norm1(mha_out+X)
        
        ffn_out = self.ffn_layer(X)
        
        X = self.norm2(X+ffn_out)
        
        return X
    
self.w_proj = nn.Linear(d_model, d_model)
self.a_proj = nn.Linear(d_model, r_model)
self.b_proj = nn.Linear(r_model, d_model)

def lora(self, X):
    # X @ W + X @ A @ B
    self.w_proj(X) + self.b_proj(self.a_proj(X))
    
    
class CacheKV(nn.module):
    def __init__(self):
        super().__init__()
        
        self.K_cache = None
        self.V_cache = None
        
    def update_cache(self, K_new, V_new):
        if self.K_cache is None:
            self.K_cache = K_new
            self.V_cache = V_new
            
        self.K_cache = torch.cat([self.K_cache, K_mew], dim=2)
        self.V_cache = torch.cat([self.V_cache, V_mew], dim = 2)
        
        
        
class GPT(nn.module):
    def __init__(self, d_model, vocab_size, max_seq_len, num_layers):
        super().__init__()        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.llm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.llm.head.weight = self.token_embedding.weight
        
    def forward(self, token_ids):
        B, T = token_ids.shape
        
        X = self.token_embedding(token_ids)
        
        postional_ids = torch.arange(T)
        
        pos = self.positional_embedding(postional_ids)
        
        X = X + pos
        
        for block in self,blocks:
            X = block(X)
            
        X = self.final_norm(X)
        
        logits = self.lm_head(X)        
        
        return logits