# %%
import os
import subprocess
import time
from multiprocessing import cpu_count
# %%
num_physical_cpus = cpu_count()
print(f"Number of physical CPUs: {num_physical_cpus}")
# You can also consider logical CPUs, but this often leads to oversubscription
num_logical_cpus = os.cpu_count()
print(f"Number of logical CPUs: {num_logical_cpus}")
# %%
import torch
from transformers import (
    AutoConfig,
    BloomForCausalLM,
)

model_org = "bigscience"
model_name = "bloom-560m"
model_id = os.path.join(model_org, model_name)
local_bin_fn = "pytorch_model.bin"
abs_fn = os.path.join("/home/antonio/text-generation-inference/server/text_generation_server/weights", model_name, local_bin_fn)

config = AutoConfig.from_pretrained(model_id)
# %%
from typing import Dict
from collections import defaultdict

def shared_pointers(tensors: Dict):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


loaded = torch.load(abs_fn, map_location="cpu")
if "state_dict" in loaded:
    loaded = loaded["state_dict"]
shared = shared_pointers(loaded)
for shared_weights in shared:
    for name in shared_weights[1:]:
        loaded.pop(name)
# For tensors to be contiguous
loaded = {k: v.contiguous() for k, v in loaded.items()}
# %%
for k, v in loaded.items():
    print(k, v.shape)
# %%
# safetensors
from safetensors import safe_open

tensors = {}
abs_fn = os.path.join("/home/antonio/text-generation-inference/server/text_generation_server/weights", model_name, "model.safetensors")
with safe_open(abs_fn, framework="pt", device=0) as f:
    for k in f.keys():
        tensor_slice = f.get_slice("word_embeddings.weight")
        vocab_size, hidden_dim = tensor_slice.shape
        tensor = tensor_slice[:, :hidden_dim]
