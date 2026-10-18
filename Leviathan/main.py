from core import Aicore
import torch

dummy = torch.rand(10,16)
model = Aicore.architecture.MLP(
    data_dim=16,output_dim=3,depth_submodel=2,sub_model_dim=(8,16),
    variant='cls'
)
result = Aicore.architecture.mlp_runner(model,dummy)
print(result)