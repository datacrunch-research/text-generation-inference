# %%
import torch

import cuda.cudart as cudart
import cuda.cuda as cuda
import os
import subprocess
from packaging.version import (
    parse,
    Version
)
from pathlib import Path
import re
import json

from dataclasses import dataclass, asdict

from tqdm import tqdm
# %%
# Verify You Have a Supported Version of Linux
raw_output = subprocess.check_output(["uname", "-m"], universal_newlines=True)
print(f"System architecture: {raw_output.strip()}")
print(os.getenv("LD_LIBRARY_PATH", None))
# %%
# Check GPU Information
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
# %%
# Check system CUDA version
err, version = cudart.cudaDriverGetVersion()
print(f"CUDA driver version: {version}")
runtime_version = cudart.cudaRuntimeGetVersion()
print(f"CUDA runtime version: {runtime_version}")
# %%
raw_output = subprocess.check_output(["nvcc", "-V"], universal_newlines=True)
output = raw_output.split()
release_idx = output.index("release") + 1
bare_metal_version = parse(output[release_idx].split(",")[0])
print(f"CUDA bare metal version: {bare_metal_version}")
# PyTorch CUDA version
print("Pytorch version\t: {}".format(torch.__version__))
print(f"System's CUDA version: {torch.version.cuda}")

print(f"GPU supported architectures: {torch.cuda.get_arch_list()}")

for i in range(torch.cuda.device_count()):
    print("GPU\t\t: {}".format(torch.cuda.get_device_name(i)))
# %%
USR_LOCAL: str = "/usr/local"
CUDA_DIR_PATH: str = os.path.join(USR_LOCAL, "cuda")
CUDA_BIN_PATH: str = os.path.join(CUDA_DIR_PATH, "bin")

torch_binary_version = parse(torch.version.cuda)
print(f"Pytorch CUDA version: {torch_binary_version}")
print(f"System's CUDA version: {bare_metal_version}")

print("\nCompiling cuda extensions with")
print(raw_output + "from " + CUDA_DIR_PATH + "/bin\n")

if (bare_metal_version != torch_binary_version):
    print(
        "Cuda extensions are being compiled with a version of Cuda that does "
        "not match the version used to compile Pytorch binaries.  "
        "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
        + "In some cases, a minor-version mismatch will not cause later errors:  "
        "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
        "You can try commenting out this check (at your own risk)."
    )

# %%
