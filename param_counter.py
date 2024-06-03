import torch
import torchvision.models as torchmodels
from models.knvision_transformer import knvit_s_16, knvit_s_8
from models.vision_transformer import vit_s_16, vit_b_16_ln, vit_s_8

model = knvit_s_16()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in knvit_s_16:", total_params)

model = knvit_s_8()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in knvit_s_8:", total_params)

model = torchmodels.vit_b_16()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in vit_b_16:", total_params)

model = torchmodels.vit_b_32()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in vit_b_32:", total_params)

model = torchmodels.vit_l_16()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in vit_l_16:", total_params)

model = vit_s_16()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in vit_s_16:", total_params)

model = vit_s_8()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in vit_s_8:", total_params)

model = vit_b_16_ln()

total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters in vit_b_16_ln:", total_params)