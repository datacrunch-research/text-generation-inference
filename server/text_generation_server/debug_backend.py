# 	SAFETENSORS_FAST_GPU=1 python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve distilgpt2
import subprocess
import os
from pathlib import Path

wd_dir: str = Path(__file__).parent.absolute()
cli_path: str = os.path.join(wd_dir, "cli.py")
os.environ["SAFETENSORS_FAST_GPU"] = "1"
sharded = "--sharded"
num_shards = 2
if num_shards > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torchrun_args = ["torchrun",
                 "--standalone",
                 "--nnodes=1",
                 f"--nproc_per_node={num_shards}",
                 ]
command: str = f"python -m torch.distributed.run --nproc_per_node=1 {cli_path} serve bigscience/bloom-560m {sharded}"
torchrun_cmd: str = f"{cli_path} serve bigscience/bloom-560m {sharded}"
torchrun_args.extend(torchrun_cmd.split())
subprocess.run(command.split())
# print(" ".join(torchrun_args))
# subprocess.run(torchrun_args)