from core.Aicore import layer
from torch import nn 
import torch
from typing import Literal

class MiniBlockMLP (nn.Module) :
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
        return nn.functional.relu(x)

class MLP (nn.Module) :
    def __init__(self,data_dim : int,output_dim : int,depth_submodel : int=1,
                sub_model_dim : tuple = (32),
                 variant : Literal['reg','bin','cls'] = 'reg') :
        super().__init__()
        self.head = layer.Linear(data_dim,16)
        if len(sub_model_dim) != depth_submodel :
            raise ValueError("len of sub_model depth must being same with depth_submodel example : dim = (32,16) depth = 2")
        model = [
            MiniBlockMLP(input_dim=16,output_dim=sub_model_dim[0])
        ]
        idx = 0 
        if depth_submodel > 1 :
            for _ in range(depth_submodel - 1) :
                model.append(MiniBlockMLP(
                    input_dim=sub_model_dim[idx],output_dim=sub_model_dim[idx+1]
                ))
                idx +=1 
        self.modelcore = nn.Sequential(*model)
        self.outlayer = layer.Linear(sub_model_dim[-1],out_dim=output_dim)

        self.varian = variant

    def forward(self,x) :
        x = self.head(x)
        x = nn.functional.celu(x)  
        x = self.modelcore(x)
        return self.outlayer(x) 

def mlp_runner (model : MLP, data ) :
    with torch.no_grad() :
        x = model(data)
        if model.varian == 'bin' :
            x = nn.functional.sigmoid(x)
        elif model.varian == 'cls' :
            x = nn.functional.softmax(x)

    return x 


