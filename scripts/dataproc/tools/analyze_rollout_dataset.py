"""
python scripts/dataproc/tools/analyze_rollout_dataset.py
"""

import os
import json
import numpy as np

# dir_path = "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter0/pi0-83pct_250430_180817_iter0_pi0_vanilla_epochs=3"
dir_path = "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter0/original_pi0-62pct_max-car-8_250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3"

reward_list = []

for data_type in ["train", "val", "test"]:
    for file in os.listdir(os.path.join(dir_path, data_type)):
        if file.endswith(".json") and "_summary" not in file and "original" not in file:
            with open(os.path.join(dir_path, data_type, file), "r") as f:
                data = json.load(f)
            reward_list.append(data[-1]["reward"])

print(f"path: {dir_path}")
print(f"Min reward: {np.min(reward_list)}")
print(f"Max reward: {np.max(reward_list)}")
