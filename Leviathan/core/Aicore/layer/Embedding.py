from torch import nn
import torch

class Embedding (nn.Module) :
    def __init__(self,vocab_size : int,embed_dim : int) :
        super().__init__()
        self.w = nn.Parameter(
            data = torch.rand(vocab_size,embed_dim)
        ) 
        nn.Embedding()

    def forward(self,x):
        return self.w[x] 