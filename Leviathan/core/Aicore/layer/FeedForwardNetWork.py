from torch import nn 


class FFN (nn.Module) :
    def __init__(self,embed_dim : int,hidden_dim : int) :
        super().__init__()
        self.layer1 = nn.Linear(embed_dim,hidden_dim,bias=False)
        self.layer2 = nn.Linear(hidden_dim,embed_dim,bias=False)

    def forward(self,x) :
        x = self.layer1(x)
        x = nn.functional.gelu(x)
        x = self.layer2(x)
        return x 