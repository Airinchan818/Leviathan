import torch 
from torch import nn 

class Attention (nn.Module) :
    def __init__(self,embedding_dim : int,use_causalmask = False) :
        super().__init__()
        self.q = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.k = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.v = nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.causalmask = use_causalmask

    def forward(self,q,k,v) :
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        score = torch.matmul(q,k.permute(0,-1,-2)) / float(k.shape[-1]**0.5)
        score = nn.functional.softmax(score,dim=-1)
        if self.causalmask is True :
            mask = 1 - torch.tril(torch.ones(size=(q.shape[1],q.shape[1])),diagonal=0).to(q.device)
            mask *= -1e9
            score += score 

        out = torch.matmul(score,v)
        return out 
    
            
         
