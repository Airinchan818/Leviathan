from torch import nn
import torch
import math  

class Linear (nn.Module) :
    def __init__(self,input_dim,out_dim) :
        super().__init__()
        std = float(math.sqrt(2 / (input_dim +out_dim)))
        self.w = nn.Parameter(data=torch.normal(mean=0.0,std=std,size=(input_dim,out_dim)))

    def forward(self,x) :
        return torch.matmul(x,self.w) 