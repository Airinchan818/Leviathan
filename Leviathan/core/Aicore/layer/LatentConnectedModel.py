from torch import nn 


class LCM (nn.Module) :
    def __init__(self,embed_dim,drop_rate : float = 0.1) :
        super().__init__()
        self.layer1 = nn.Linear(embed_dim,embed_dim,bias=False)
        self.layer2 = nn.Linear(embed_dim,embed_dim,bias=False)
        self.layer3 = nn.Linear(embed_dim,embed_dim,bias=False)
        self.norm = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward (self,x) :
        z = self.norm(z)
        step1 = self.layer1(z)
        step1 = nn.functional.gelu(step1)

        step2 = self.layer2(z)
        step2 = nn.functional.gelu(step2)

        x = step1 + step2 
        x = self.dropout(x)
        x = self.layer3(x)
        x = nn.functional.tanh(x)
        return x 
        