import torch 
from torch import nn 
from typing import Literal 

class LayerScale (nn.Module) :
    def __init__(self,embed_dim,factor : Literal['a','b'] = 'b') :
        super().__init__()
        self.alpha = nn.Parameter(
            data=torch.ones(size=(embed_dim,))
        )
        self.factor = factor 

    def forward(self,x,y) :
        if self.factor == 'a' :
            return (x * self.alpha) + y
        else :
            return x + (self.alpha * y)
        