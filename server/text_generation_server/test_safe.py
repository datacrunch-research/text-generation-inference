import os
from safetensors import safe_open
import torch

model_org = "bigscience"
model_name = "bloom-560m"
model_id = os.path.join(model_org, model_name)
local_bin_fn = "pytorch_model.bin"


tensors = {}
sf_filename = os.path.join("/home/antonio/text-generation-inference/server/text_generation_server/weights", model_name, "model.safetensors")
pt_filename = os.path.join("/home/antonio/text-generation-inference/server/text_generation_server/weights", model_name, local_bin_fn)

weights = {}
with safe_open(sf_filename, framework="pt", device="cuda:0") as f:
    for k in f.keys():
        weights[k] = f.get_slice(k)[:1]
        #print(k, weights[k].shape)
tweights = torch.load(pt_filename, map_location="cuda:0")
tweights = {k: v[:1] for k, v in tweights.items()}
for k, v in weights.items():
    tv = tweights[k]
    print(f"Safetensors: {k}, {v}")
    print(f"Pytorch: {k}, {tv}")
