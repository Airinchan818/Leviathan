import torch 
from torch import nn 

class RMSnorm (nn.Module) :
    def __init__(self,embed_dim,eps=1e-7) :
        super().__init__()
        self.gamma = nn.Parameter(
            data=torch.ones(size=(embed_dim,))
        )
        self.eps = eps 

    def forward(self,x) :
        rms = torch.sqrt(
            torch.mean(x**2,dim=-1,keepdim=True)
        )
        x = x /rms 
        return x * self.gamma