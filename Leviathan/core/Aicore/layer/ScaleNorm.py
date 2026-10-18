import torch 
from torch import nn 

class ScaleNorm (nn.Module) :
    def __init__(self,embed_dim : int,eps = 1e-6) :
        super().__init__()
        self.alpha = nn.Parameter(
            data=torch.ones(size=(embed_dim,))
        )
        self.eps = eps 

    def forward(self,x) :
        sumval = torch.sum((x*x),dim=-1,keepdim=True)
        norm = torch.sqrt(sumval + self.eps)
        return x * (self.alpha / norm) 