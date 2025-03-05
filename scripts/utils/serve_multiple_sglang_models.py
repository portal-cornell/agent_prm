"""
Usage (from the root directory):
python scripts/utils/serve_multiple_sglang_models.py

When to use it, if you want to serve multiple SGLang models on the same machine.
"""
import hydra
from omegaconf import DictConfig
from agent_prm.utils.general_utils import start_sglang_server

@hydra.main(config_path="../../configs", config_name="sglang_servers.yaml", version_base=None)
def main(cfg: DictConfig):
    for server_config in cfg.server_configs:
        print(f"\nStarting SGLang server on port {server_config.server_url}")
        port = int(server_config.server_url.split(":")[-1][:-1])
        _, _, _ = start_sglang_server(model_path=server_config.model_id,
                                            port=port,
                                            tp=server_config.tp,
                                            dist_url_port=server_config.dist_url_port,
                                            gpu_id=server_config.gpu_id)
        
    print(f"\nAll SGLang servers have been started")
    while True:
        print("Press Enter to exit")
        input()
        break
                                            
if __name__ == "__main__":
    main()