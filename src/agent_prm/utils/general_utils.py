import json
import torch
import subprocess
import os
import atexit
import signal
import sys
from sglang.utils import wait_for_server

def load_json(fp: str):
    with open(fp, "r") as f:
        return json.load(f)
    
def save_json(fp: str, data: dict):
    with open(fp, "w") as f:
        json.dump(data, f, indent=4)

def start_sglang_server(model_path, port, tp=1, dist_url_port=29500, gpu_id=None):
    """
    Starts an SGLang server on the given port.

    We reserve the GPUs with the highest IDs for SGLang. For example,
    for 4 GPUs and tp=2, we reserve GPUs 2 and 3 for SGLang.

    If you are running this to evaluate during training, you must set
    dist_url to the same MASTER_ADDR:MASTER_PORT as the training script.

    Args:
        model_path (str): The model to use for code generation.
        port (int): The port to run the server on.
        tp (int): The number of GPUs to use for tensor parallelism.
        dist_url (str): The URL for distributed training.
    """
    num_gpus = torch.cuda.device_count()
    if gpu_id is None:
        base_gpu_id = num_gpus - tp
    else:
        base_gpu_id = gpu_id

    env = os.environ.copy()

    session_id = None
    if env.get("TMUX") is not None:
        session_id = env.get("TMUX").split(",")[-1]

    env['OUTLINES_CACHE_DIR'] = f"~/.cache/outlines{'_' + str(session_id) if session_id is not None else ''}_{base_gpu_id}"

    print(f"OUTLINES_CACHE_DIR: {env['OUTLINES_CACHE_DIR']}")

    command = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dist-init-addr", f"localhost:{dist_url_port}",
        "--base-gpu-id", str(base_gpu_id), # Reserve highest ID GPUs for SGLang
        "--tp", str(tp)
    ]
    print (command)
    # Start the process in a new session
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # Start a new session
        env=env
    )

    # Ensure the process group is terminated when the script exits
    def cleanup():
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            print(f"Process {process.pid} has already been terminated")
            # Process has already been terminated
            return
    atexit.register(cleanup)

    # Wait for the server to start
    base_url = f"http://0.0.0.0:{port}"
    print(f"Starting SGLang server on {base_url}")
    wait_for_server(base_url)
    print(f"SGLang server has started on {base_url}")

    return process, f"{base_url}/v1", base_gpu_id