from torch import nn 
from typing import Literal 
import torch 

class LogitMixingAttention (nn.Module) :
    def __init__ (self,embed_dim,use_causalmask : bool = False,
                  mode : Literal['kfactor','scale'] = 'kfactor') :
        super().__init__()
        self.lg = nn.Linear(embed_dim,embed_dim)
        self.lo = nn.Linear(embed_dim,embed_dim)
        self.alpha = nn.Parameter(
            torch.zeros(size=(1,1,embed_dim))
        )
        self.mode = mode 
        self.causal_mask = use_causalmask

    def kfactor (self,q,k,v) :
        state = q + (self.alpha * k ) / float(k.shape[-1]**0.5)
        v = self.lg(v)
        score = nn.functional.softmax(state,dim=-1)
        score = (score * v ) + v 
        out = self.lo(score)
        return out 

    def scaled (self,q,k,v) :
        b,s,d = q.shape
        state = q + (self.alpha * (k + v))
        lg = self.lg(state)
        score = torch.matmul(lg,k.permute(0,-1,-2)) / float(k.shape[-1]**0.5)
        score = nn.functional.softmax(score)
        if self.causal_mask is True :
            mask = 1 - torch.tril(torch.ones(size=(s,s)),diagonal=0) 
            mask *= -1e9 
            score = score + mask 
        out = torch.matmul(score,out)
        return out 
     

