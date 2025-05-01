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
import shutil
from sglang.utils import wait_for_server

SGLANG_LOCAL_DIR = "/share/portal/hw575/agent_prm/_cached_models"

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


def start_sglang_server(model_path, port, tp=1, dist_url_port=29500, gpu_id=None, max_try=10, local_sglang: bool = False):
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

            base_cache_dir = os.path.join(base_cache_dir, f"{base_gpu_id}")

            if os.path.exists(base_cache_dir):
                print(f"Removing existing cache dir {base_cache_dir}")
                # Remove the cache dir
                shutil.rmtree(base_cache_dir)

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
            if local_sglang:
                command.append("--download-dir")
                command.append(SGLANG_LOCAL_DIR)
                os.makedirs(SGLANG_LOCAL_DIR, exist_ok=True)

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


def setup_sglang_server(agent_config: dict, local_sglang: bool = False):
    """
    Setup the SGLang server

    If the agent is a SGLang server, we only need to start one server
    If the agent is a Best of N, we need to start two servers
        - One for the generator
        - One for the critic

    Returns:
        a list of processes
    """
    processes = []
    if agent_config.type == "sglang_server":
        if "TODO" in agent_config.server_url:
            port = None
        else:
            port = int(agent_config.server_url.split(":")[-1][:-1])

        print(f"Starting SGLang server on port {port}")
        process, server_url, _ = start_sglang_server(model_path=agent_config.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.dist_url_port,
                                                local_sglang=local_sglang)
        
        if "TODO" in agent_config.server_url:
            agent_config.server_url = server_url

        processes.append(process)
    elif agent_config.type == "dual_sglang_server_agents":
        # Start the api caller
        if "TODO" in agent_config.api_caller.server_url:
            port = None
        else:
            port = int(agent_config.api_caller.server_url.split(":")[-1][:-1])

        print(f"Starting SGLang server for the api caller on port {port}, serving on the highest ID GPU")
        process, server_url, base_gpu_id = start_sglang_server(model_path=agent_config.api_caller.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.api_caller.dist_url_port,
                                                local_sglang=local_sglang)
        
        if "TODO" in agent_config.api_caller.server_url:
            agent_config.api_caller.server_url = server_url

        processes.append(process)
        
        # Start the response generator
        gpu_id = max(0, base_gpu_id - 1)
        if "TODO" in agent_config.response_generator.server_url:
            port = None
        else:
            port = int(agent_config.response_generator.server_url.split(":")[-1][:-1]) 

        print(f"Starting SGLang server for the response generator on port {port}, serving on the next highest ID GPU {gpu_id}")
        process, server_url, _ = start_sglang_server(model_path=agent_config.response_generator.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.response_generator.dist_url_port,
                                                gpu_id=gpu_id,
                                                local_sglang=local_sglang)
            
        if "TODO" in agent_config.response_generator.server_url:
            agent_config.response_generator.server_url = server_url

        processes.append(process)
    elif agent_config.type == "best_of_n" or agent_config.type == "sglang_server_with_critic":
        # Start the critic
        if "TODO" in agent_config.critic.server_url:
            port = None
        else:
            port = int(agent_config.critic.server_url.split(":")[-1][:-1])

        print(f"Starting SGLang server for the critic on port {port}, serving on the highest ID GPU")
        process, server_url, base_gpu_id = start_sglang_server(model_path=agent_config.critic.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.critic.dist_url_port,
                                                local_sglang=local_sglang)
        
        if "TODO" in agent_config.critic.server_url:
            agent_config.critic.server_url = server_url

        processes.append(process)

        # Start the generator
        #    One condition to not host sglang: if generator has the field host_sglang and it is False
        gpu_id = max(0, base_gpu_id - 1)
        if agent_config.type == "best_of_n":
            if not (hasattr(agent_config.generator, 'host_sglang') and not agent_config.generator.host_sglang):
                if "TODO" in agent_config.generator.server_url:
                    port = None
                else:
                    port = int(agent_config.generator.server_url.split(":")[-1][:-1])

                print(f"Starting SGLang server for the generator on port {port}, serving on the next highest ID GPU {gpu_id}")
                process, server_url, _ = start_sglang_server(model_path=agent_config.generator.model_id,
                                                        port=port, 
                                                        tp=1,
                                                        dist_url_port=agent_config.generator.dist_url_port,
                                                        gpu_id=gpu_id,
                                                        local_sglang=local_sglang)
                
                if "TODO" in agent_config.generator.server_url:
                    agent_config.generator.server_url = server_url
                
                processes.append(process)
            else:
                print(f"Generator hosted sglang at {agent_config.generator.server_url}")
        elif agent_config.type == "sglang_server_with_critic":
            if "TODO" in agent_config.server_url:
                port = None
            else:
                port = int(agent_config.server_url.split(":")[-1][:-1])

            print(f"Starting SGLang server for the critic on port {port}, serving on the highest ID GPU")
            process, server_url, _ = start_sglang_server(model_path=agent_config.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.dist_url_port,
                                                gpu_id=base_gpu_id,
                                                local_sglang=local_sglang)
            
            if "TODO" in agent_config.server_url:
                agent_config.server_url = server_url

            processes.append(process)

    return processes