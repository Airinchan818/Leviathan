from torch import nn 
import torch 
from core.Aicore import layer 
from typing import Literal 

godevice = "cuda" if torch.cuda.is_available() else "cpu"

class CATDecoderBlock (nn.Module) :
    def __init__(self,embed_dim : int,hidden_dim : int,
                norm_var : Literal['rms','norm','scale']='rms',
                method : Literal['pre','post'] = 'pre',
                res_method : Literal['add','scale','zeros','random'] = 'add',
                use_causalmask = False) :
        super().__init__()
        self.attention1 = layer.Attention(embed_dim,use_causalmask)
        self.ffn = layer.FFN(embed_dim,hidden_dim)
        self.methode = method
        self.attention2 = layer.Attention(embed_dim,use_causalmask)
        if norm_var == 'norm' :
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.norm3 = nn.LayerNorm(embed_dim)
        elif norm_var == 'scale' :
            self.norm1 = layer.ScaleNorm(embed_dim)
            self.norm2 = layer.ScaleNorm(embed_dim)
            self.norm3 = layer.ScaleNorm
        elif norm_var == 'rms'  :
            self.norm1 = layer.RMSNorm(embed_dim)
            self.norm2 = layer.RMSNorm(embed_dim)
            self.norm3 = layer.RMSNorm(embed_dim)

        if res_method == 'random' :
            self.res_modul1 = layer.ReRandom(embed_dim)
            self.res_modul2 = layer.ReRandom(embed_dim)
            self.res_modul3 = layer.ReRandom(embed_dim)
        elif res_method == 'zeros' :
            self.res_modul1 = layer.ReZero(embed_dim)
            self.res_modul2 = layer.ReZero(embed_dim)
            self.res_modul3 = layer.ReZero(embed_dim)
        elif res_method =='scale' :
            self.res_modul1 = layer.LayerScale(embed_dim)
            self.res_modul2 = layer.LayerScale(embed_dim)
            self.res_modul3 = layer.LayerScale(embed_dim)

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.res_method = res_method

    def forward(self,x,y = None) :
        if self.methode == 'pre' :
            z = self.norm1(x)
            z = self.attention1(z,z,z)
            z = self.drop1(z)
            if self.res_method == 'add' :
                x = x + z 
            else :
                x = self.res_modul1(x)

            if y is not None :
                z = self.norm2(y)
                z = self.attention(x,z,z)
                z = self.drop2(z)
                if self.res_method == 'add':
                    x = x + z
                else :
                    x = self.res_modul2(x,z)

            z = self.norm3(x)
            z = self.ffn(z)
            z = self.drop3(z)

            if self.res_method == 'add' :
                x = x + z
            else :
                x = self.res_modul3(x,z)

            return x 

        else :
            z = self.attention1(x,x,x)
            z = self.drop1(z)
            z = self.norm1(z)

            if self.res_method == 'add' :
                x = x + z 
            else :
                x = self.res_modul1(x,z)

            if y is not None :
                z = self.attention2(x,y,y)
                z = self.drop2(z)
                z = self.norm2(z)

                if self.res_method == 'add' :
                    x = x + z 
                else :
                    x = self.res_modul2(x,z)

            z = self.ffn(x)
            z = self.drop3(z)
            z = self.norm3(z)

            if self.res_method == 'add' :
                x = x + z 
            else :
                x = self.res_modul3(x,z)
            
            return x 
              
class CATDecoder (nn.Module) : 
    def __init__(self,vocab_size : int,embed_dim : int,hidden_dim : int,
                model_depth : int = 2,maxpos : int = 10,norm_var : Literal['rms','norm','scale']='rms',
                method : Literal['pre','post'] = 'pre',
                res_method : Literal['add','scale','zeros','random'] = 'add',
                use_causalmask = False)  :
        super().__init__()
        self.embedding = layer.Embedding(vocab_size,embed_dim)
        self.embed_pos = layer.Embedding(maxpos,embed_dim)
        self.scale = float(embed_dim **0.5)
        self.modelcore = nn.ModuleList([
            CATDecoderBlock(embed_dim,hidden_dim,norm_var,method,
                            res_method,use_causalmask) for _ in range(model_depth)
        ])
        self.to(godevice)

    def forward(self,x,y = None) :
        b,s = x.shape 
        x = self.embedding(x)
        x = x * self.scale 
        pos = torch.arange(0,s).to(godevice)
        pos = self.embed_pos(pos)
        pos = torch.unsqueeze(pos,dim=0)
        x = x + pos 
        for layer in self.modelcore :
            x = layer(x)
        return x 

def CATDecoder_runner (model : CATDecoder,datax : torch.Tensor,datay = None,
                       device : Literal['cpu','cuda'] = None ) :
    if model.training is True :
        model.eval()

    target_device = device if device is not None else godevice
    if str(next(model.parameters()).device) != target_device :
        model.to(target_device)

    if str(datax.device) != target_device :
        datax.to(target_device)
    
    if datay is not None :

        if str(datay.device) != target_device :
            datay.to(target_device)

    with torch.no_grad() :
        x = model(datax,datay)
    return x 

    