from torch  import nn
import torch 
from core.Aicore import architecture as ar 
import pandas as pd 

def model_loader (path : str,class_model : nn.Module) :
    model = class_model.load_state_dict(torch.load(path))
    return model

def getting_model_name (model_name : str) :
    model_name = pd.read_json("core/Aicore/indexmodel/" + model_name) 
    print(f"Model Details :\n {model_name}")
    return model_name 

def model_naming (class_model : nn.Module, name : str) :
    

class ModelNode  (nn.Module):
    def __init__(self,model : nn.Module, contract_list : list,contact_action : list) :
        self.model = model 
        self.name = model._get_name()
        contract_model_name = []
        for name in contract_list :
            model_name = getting_model_name(name)


