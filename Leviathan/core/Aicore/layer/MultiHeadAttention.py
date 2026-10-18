import torch 
from torch import nn 

class MultiHeadAttention (nn.Module) :
    def __init__(self,embedding_dim : int,num_head : int,use_causal_mask :bool= False) :
        super().__init__()
        self.q = nn.Linear(embedding_dim,embedding_dim)
        self.k = nn.Linear(embedding_dim,embedding_dim)
        self.v = nn.Linear(embedding_dim,embedding_dim)
        self.use_causalmask = use_causal_mask 
        self.num_head = num_head 
        self.div_dim = embedding_dim // num_head

    def __split_head(self,x) : 
        b,s,_ = x.shape 
        x = torch.reshape(b,s,self.num_head,self.div_dim)
        return torch.permute(0,2,1,3)

    def forward(self,q,k,v) :
        b,s,d= q.shape
        q = self.q(q)
        k= self.k(k)
        v = self.v(v)
        q = self.__split_head(q)
        k = self.__split_head(k)
        v = self.__split_head(v)
        
        score = torch.matmul(q,k.permute(0,-3,-1,-2)) / float(self.div_dim)
        score = nn.functional.softmax(score,dim=-1)

        if self.use_causalmask is True :
            mask = 1 - torch.tril(torch.ones(size=(s,s)),diagonal=0)
            mask *= -1e9
            score += mask 

        out = torch.matmul(score,v)
        out = out.permute(0,-3,-1,-2)
        out = out.reshape(b,s,d)
        return out  
        

