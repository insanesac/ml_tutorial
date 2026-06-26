from turtle import pos
import numpy as np
import math
import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, d_model, vocab_size, max_seq_len, num_layers, num_heads):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        
        self.llm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.llm_head.weight =  self.token_embedding.weight
        
    def forward(self, token_ids):
        B, T = token_ids.shape
        
        X = self.token_embedding(token_ids)
        
        positonal_ids = torch.arange(T)
        
        pos = self.positional_embedding(positonal_ids)
        
        X = X + pos
        
        for block in self.blocks:
            X = block(X)
            
        X = self.final_norm(X)
        
        logits = self.llm_head(X)

        return logits            
    
    
we use ModuleList to make the transformer blocks a trainable set 
weight tying is used so that the input and output shares the same semantic space. 
(B, T, D)
(T,)
we use this so that attention can also get positional information of each token, without this "I am cat" and "Cat am I" are semantically similar
we only computer the positional enbedding of token 2049
