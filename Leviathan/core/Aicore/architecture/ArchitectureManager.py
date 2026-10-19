import torch 
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

def delete_model(model : torch.nn.Module) :
    del model
    gc.collect()
    if device == "cuda" :
        torch.cuda.empty_cache()

    return None 
    