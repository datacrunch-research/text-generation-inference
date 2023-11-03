# 	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve distilgpt2
import subprocess
import os
from pathlib import Path


wd_dir: str = Path(__file__).parent.absolute()
cli_path: str = os.path.join(wd_dir, "cli.py")
os.environ["SAFETENSORS_FAST_GPU"] = "1"
model_org = "bigscience"
model_name = "bloom-560m"
model_id = os.path.join(model_org, model_name)
abs_fn = os.path.join("/home/antonio/text-generation-inference/server/text_generation_server/weights", model_name)
# torch.distributed configuration
os.environ["OMP_NUM_THREADS"] = "16"
sharded = "--sharded"
num_shards = 2
if num_shards > 1 and sharded:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.distributed.launch (old mechanism)
logger_level: str = "DEBUG"
# if dtype is None: dtype = torch.float16
dtype: str = "float16"
checkpoint_ext: str = ".bin"
serve_args = f"serve {abs_fn} {sharded} --dtype {dtype} --logger-level {logger_level} --checkpoint-ext {checkpoint_ext}"
command: str = f"python -m torch.distributed.run --nproc_per_node={num_shards} {cli_path}" + " " + serve_args
print(command)
subprocess.run(command.split())

"""
torchrun (elastic launch)

refs:
    [TORCHRUN (ELASTIC LAUNCH)](https://pytorch.org/docs/stable/elastic/run.html)

torchrun_args = ["torchrun",
                 "--standalone",
                 "--nnodes=1",
                 f"--nproc_per_node={num_shards}",
                 ]
logger_level: str = "DEBUG"
serve_args = f"serve {model_id} {sharded} --logger-level {logger_level}"
torchrun_cmd: str = f"{cli_path}" + " " + serve_args
print(torchrun_cmd)
torchrun_args.extend(torchrun_cmd.split())
print(" ".join(torchrun_args))
subprocess.run(torchrun_args)
"""