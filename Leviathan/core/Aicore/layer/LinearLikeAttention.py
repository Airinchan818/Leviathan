from torch import nn 
import torch 

class LinearLikeAttention (nn.Module) :
    def __init__(self,input_dim : int ,output_dim : int , num_split : int = None) :
        super().__init__()
        self.split = num_split if num_split is not None else input_dim // 8 
        self.q = nn.Linear(input_dim//self.split,output_dim//self.split)
        self.k = nn.Linear(input_dim//self.split,output_dim//self.split)
        self.v = nn.Linear(input_dim//self.split,output_dim//self.split)
        self.out_dim = output_dim 

    def forward(self,q,k,v) :
        b,d = q.shape
        d = d // self.split
        q = torch.reshape(q,shape=(b,self.split,d)) 
        k = torch.reshape(k,shape=(b,self.split,d)) 
        v = torch.reshape(v,shape=(b,self.split,d)) 
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        qk = torch.matmul(q,k.permute(0,-1,-2)) / float(k.shape[-1] **0.5)
        qk = nn.functional.softmax(qk,dim=-1)
        out = torch.matmul(qk,v)
        out = torch.reshape(out,shape=(b,self.out_dim))
        return out 

