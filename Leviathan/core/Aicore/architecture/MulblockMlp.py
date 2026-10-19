from torch import nn 
from core.Aicore import layer
import torch 
from typing import Literal
godevice = "cuda" if torch.cuda.is_available() else "cpu"

class MiniBlock(nn.Module) :
    def __init__(self,input_dim : int ,output_dim : int ,expand_dim : int = 2 ) :
        super().__init__()
        start_dim = 4 * expand_dim 
        self.layer1 = layer.Linear(input_dim,start_dim)
        start_dim2= start_dim * expand_dim
        self.layer2 = layer.Linear(start_dim,start_dim2)
        self.layer3 = layer.Linear(start_dim2,output_dim)

    def forward(self,x) :
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        return x 



class MultiBlockMLP (nn.Module) :
    def __init__(self,data_dim : int,output_dim : int,depth_submodel : int=2,
                sub_model_dim : int = 32,
                 variant : Literal['reg','bin','cls'] = 'reg',
                 norm : Literal['scale','rms','norm'] = 'rsm',
                 res_var : Literal['random','scale','zero'] ='random')  :
        super().__init__()
        self.head = layer.Linear(data_dim,16)
        res_block = []
        for _ in range(depth_submodel) :
            if  res_var == 'random' :
                res_block.append(layer.ReRandom(sub_model_dim))
            elif res_var == 'scale' :
                res_block.append(layer.LayerScale(sub_model_dim))
            else :
                res_block.append(layer.ReZero(sub_model_dim))

        self.res_block = nn.ModuleList(res_block)

        model_block = []
        for _  in range(depth_submodel) :
            model_block.append(MiniBlock(16,sub_model_dim))
        self.model_block = nn.ModuleList(model_block)

        if norm == 'rms' : 
            self.norm = layer.RMSnorm(sub_model_dim)
        elif norm == 'norm' :
            self.norm = nn.LayerNorm(sub_model_dim)
        else :
            self.norm = layer.ScaleNorm(sub_model_dim)

        self.variant = variant
        self.out = layer.Linear(sub_model_dim,output_dim)
        self.model_depth = depth_submodel

    def forward(self,x) :
        x = self.head(x)
        y = torch.zeros_like(x) + 1e-6
        for i in range(self.model_depth) :
            y_factor = self.model_block(x)
            y = y + self.res_block(y,y_factor)  

        x = self.norm(y)
        x = self.out(x)
        return x 

def multimlp_runner (model : MultiBlockMLP,data : torch.Tensor, device : Literal['cpu','cuda'] = None ) :
    
    if model.training is True :
        model.eval()
    target_device = device if device is not None else godevice
    if str(next(model.parameters()).device) != target_device :
        model.to(target_device)
    if str(data.device) != target_device :
        data = data.to(target_device)
    with torch.no_grad() :
        x = model(data)
        if model.variant == 'reg' :
            x = x
        elif model.variant == 'bin' :
            x = nn.functional.sigmoid(x)
        elif model.variant == 'cls' :
            x = nn.functional.softmax(x)

    return x 