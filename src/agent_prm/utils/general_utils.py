import json
import torch
import subprocess
import os
import atexit
import signal
import sys
import socket
import time
import random
from sglang.utils import wait_for_server

def load_json(fp: str):
    with open(fp, "r") as f:
        return json.load(f)
    
def save_json(fp: str, data: dict):
    with open(fp, "w") as f:
        json.dump(data, f, indent=4)

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Let OS pick an available port
        return s.getsockname()[1]


def start_sglang_server(model_path, port, tp=1, dist_url_port=29500, gpu_id=None, max_try=10):
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
    try_count = 0
    while try_count < max_try:
        try:
            if port is None:
                port = get_free_port()
                print(f"Using randomly grabbed port {port}")
            
            if dist_url_port is None:
                dist_url_port = get_free_port()
                print(f"Using randomly grabbed dist_url_port {dist_url_port}")

            num_gpus = torch.cuda.device_count()
            if gpu_id is None:
                base_gpu_id = num_gpus - tp
            else:
                base_gpu_id = gpu_id

            env = os.environ.copy()

            session_id = None
            if env.get("TMUX") is not None:
                session_id = env.get("TMUX").split(",")[-1]

            base_cache_dir = os.path.join("~", ".cache", f"outlines{'_' + str(session_id) if session_id is not None else ''}")

            # os.makedirs(base_cache_dir, exist_ok=True)

            # add a random number to the cache dir (between 0 and 100)
            base_cache_dir = os.path.join(base_cache_dir, f"{base_gpu_id}_{random.randint(0, 100)}")
            env['OUTLINES_CACHE_DIR'] = base_cache_dir

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
            print(command)
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

            return process, base_url, base_gpu_id
        except Exception as e:
            print(f"Failed to start SGLang server on {base_url}: {e}")
            try_count += 1
            time.sleep(1)

    raise Exception("Failed to start SGLang server")
