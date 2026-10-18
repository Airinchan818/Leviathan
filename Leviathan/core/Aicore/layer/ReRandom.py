from torch import nn
import torch
from typing import Literal

class ReRandom (nn.Module) :
    def __init__(self,embed_dim : int,init_ : float = 1e-4,
                 factor : Literal['a','b']='b') :
        super().__init__()
        self.factors = nn.Parameter(
            data = torch.rand((embed_dim,)) * init_
        )
        self.factor = factor

    def forward(self,x,y) :
        if self.factor == 'b' :
            return x + (y * self.factors)
        else : 
            return (x * self.factors) + y