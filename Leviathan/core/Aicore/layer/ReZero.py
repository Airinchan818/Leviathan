import torch
from torch import nn 

class ReZero (nn.Module) :
    def __init__(self,embed_dim : int) :
        super().__init__()
        self.factor = nn.Parameter(
            data=torch.zeros(size=(embed_dim,))
        )

    def forward(self,x,y) :
        return x + self.factor * y
