import os
import json
import shutil
from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT

START_ROLLOUT_DIR = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"

END_ROLLOUT_DIR = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/hindsight_pi0-all-data-3epoches_250307_212417_peft=false_epoch3+all"

ROLLOUT_IDX_TO_COPY = 4

def is_valid_rollout(f: str) -> bool:
    """
    Check if the rollout is valid
    """
    return f.endswith(".json") and not f.endswith("_original.json") and not f.endswith("_summary_dict.json")

def is_within_valid_range(file_name: str, rollout_idx_min: int, rollout_idx_max: int) -> bool:
    """
    Check if the rollout idx is within the valid range
    """
    rollout_idx = int(file_name.split("_")[-1].split(".")[0])

    return rollout_idx >= rollout_idx_min and rollout_idx < rollout_idx_max

os.makedirs(END_ROLLOUT_DIR, exist_ok=True)

for data_type in ["train", "val"]:
    os.makedirs(os.path.join(END_ROLLOUT_DIR, data_type), exist_ok=True)

    if data_type == "train":
            object_dict_to_use = TRAIN_OBJECT_DICT
    elif data_type == "val":
        object_dict_to_use = VALIDATION_OBJECT_DICT

    objects_to_eval_on = [(obj, data_type) for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]

    # Get all the rollout files
    json_files = [f for f in os.listdir(os.path.join(START_ROLLOUT_DIR, data_type)) if is_valid_rollout(f)]

    # For each task, we sample N rollouts
    for obj, _ in objects_to_eval_on:
        # Get the task specific rollout files
        task_rollout_files = [f for f in json_files if obj in f and is_within_valid_range(f, 0, ROLLOUT_IDX_TO_COPY)]
        
        # Copy all these files to the end rollout dir
        for f in task_rollout_files:
            shutil.copy(os.path.join(START_ROLLOUT_DIR, data_type, f), os.path.join(END_ROLLOUT_DIR, data_type, f))

    # Copy over the summary dict
    with open(os.path.join(START_ROLLOUT_DIR, data_type, "_summary_dict.json"), "r") as f:
        summary_dict = json.load(f)

    new_summary_dict = {str(i): summary_dict[str(i)] for i in range(ROLLOUT_IDX_TO_COPY)}

    # Only save the summary dict for the sampled rollouts
    with open(os.path.join(END_ROLLOUT_DIR, data_type, "_summary_dict.json"), "w") as f:
        json.dump(new_summary_dict, f, indent=4)

    # Also save another copy for actually completing the rollout
    with open(os.path.join(END_ROLLOUT_DIR, data_type, "_rollout_complete_summary_dict.json"), "w") as f:
        json.dump(new_summary_dict, f, indent=4)

