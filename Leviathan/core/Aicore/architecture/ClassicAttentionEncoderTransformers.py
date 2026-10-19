import torch 
from torch import nn 
from typing import Literal
from core.Aicore import layer
godevice = "cuda" if torch.cuda.is_available() else "cpu"

class CATEncoderBlock (nn.Module) :
    def __init__(self,embed_dim : int,hidden_dim : int,
                norm_var : Literal['rms','norm','scale']='rms',
                method : Literal['pre','post'] = 'pre',
                res_method : Literal['add','scale','zeros','random'] = 'add') :
        super().__init__()
        self.methode = method 

        self.attention = layer.Attention(embedding_dim=embed_dim)
        self.ffn = layer.FFN(embed_dim,hidden_dim=hidden_dim)
        if norm_var == 'rms' :
            self.norm1 = layer.RMSnorm(embed_dim)
            self.norm2 = layer.RMSnorm(embed_dim)
        elif norm_var == 'norm' :
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        else :
            self.norm1 = layer.ScaleNorm(embed_dim)
            self.norm2 = layer.ScaleNorm(embed_dim)

        if res_method == 'scale' :
            self.resmodule1 = layer.LayerScale(embed_dim)
            self.resmodule2 = layer.LayerScale(embed_dim)
        elif res_method == 'random' :
            self.resmodule1 = layer.ReRandom(embed_dim)
            self.resmodule2 = layer.ReRandom(embed_dim)
        elif res_method == 'zeros' :
            self.resmodule1 = layer.ReZero(embed_dim)
            self.resmodule2 = layer.ReZero(embed_dim)

        self.dropout1 = nn.Dropout(p=0.1,)
        self.dropout2 = nn.Dropout(p=0.1)
        self.resmethod = res_method
        

    def forward(self,x) :
        if self.methode == 'pre'  :
            z = self.norm1(x)
            z = self.attention(z,z,z)
            z = self.dropout1(z)
            if self.resmethod == 'add' :
                x = x + z 
            else :
                x = self.resmodule1(x,z)

            z = self.norm2(x)
            z = self.ffn(z)
            z = self.dropout2(z)
            if self.resmethod == 'add' :
                x = x + z 
            else :
                x = self.resmodule2(x,z)

            return x 

        else :
            z = self.attention(x,x,x)
            z = self.dropout1(z)
            z = self.norm1(z)
            if self.resmethod == 'add' :
                x = x + z 
            else :
                x = self.resmodule1(x,z)

            z = self.ffn(x)
            z = self.dropout2(z)
            z = self.dropout2(z)

            if self.resmethod == 'add' :
                x = x + z 
            else :
                x = self.resmodule2(x,z)

            return x 
        

class CATEncoder (nn.Module) :
    def __init__(self,vocab_size,embed_dim : int,hidden_dim : int,model_depth,maxpos : int,
                norm_var : Literal['rms','norm','scale']='rms',
                method : Literal['pre','post'] = 'pre',
                res_method : Literal['add','scale','zeros','random'] = 'add')  :
        super().__init__()
        self.embedding = layer.Embedding(vocab_size,embed_dim) 
        self.scale = float(embed_dim **0.5)
        self.modelcore = nn.ModuleList([
            CATEncoderBlock(
                embed_dim,hidden_dim,norm_var,method,res_method
            ) for _ in range(model_depth)
        ])
        self.posEmbedding = layer.Embedding(maxpos,embed_dim)

        self.to(godevice)

    def forward(self,x) :
        _,s = x.shape
        x = self.embedding(x)
        x = x * self.scale
        pos = torch.arange(0,s).to(godevice)
        pos = self.posEmbedding(pos)
        pos = torch.unsqueeze(pos,dim=0)
        x = x + pos 
        for layer in self.modelcore :
            x = layer(x)

        return x 
   

def CTAEncoder_runner (model : CATEncoder, data : torch.Tensor , device : Literal['cpu','cuda'] = None ) :
    
    if model.training is True :
        model.eval()
    target_device = device if device is not None else godevice
    if str(next(model.parameters()).device) != target_device :
        model.to(target_device)
    if str(data.device) != target_device :
        data = data.to(target_device)
    with torch.no_grad() :
        x = model(data)
    return x  
