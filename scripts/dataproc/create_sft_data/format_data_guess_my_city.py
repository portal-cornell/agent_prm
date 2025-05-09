"""
Create training data for twenty questions

We use the following template for the data

SFT format:
[
    {
        "prompt": "string: the prompt for the training data",
        "response": "string: the response for the training data"
    }
]

Usage:
- Vanilla mode (only training from the gpt4o rollouts):

python scripts/dataproc/create_sft_data/format_data_guess_my_city.py -m vanilla -i 0
- Multi-Star mode (only training from the successful rollouts):

python scripts/dataproc/create_sft_data/format_data_guess_my_city.py -m multi-star -i 1
- LEAP mode (training from successful rollouts + rollouts with expert relabeled actions):

python scripts/dataproc/create_sft_data/format_data_guess_my_city.py -m leap -i 1
"""
import math
import json
import os
import argparse
import yaml
import random
from typing import List
from tqdm import tqdm
from jinja2 import Template
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.envs.guess_my_city.data import get_default_city_list

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw Guess My City logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/guess_my_city.yaml", help='Path to Guess My City dataproc config file')
    parser.add_argument('-m', "--mode", type=str, default="vanilla", choices=["vanilla", "leap", "multi-star"])
    parser.add_argument('-i', type=int, required=True, help='The iteration number of the rollout')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return args, cfg

def get_successful_rollouts(json_files: List[str], rollout_per_task: int) -> List[str]:
    """
    Returns:
        - a list of successful rollout paths
    """
    successful_rollouts = []
    
    obj_dict = {}

    for file in tqdm(json_files, desc="Processing rollouts"):
        obj_name = file.split("/")[-1].split("_")[0]

        if obj_name not in obj_dict:
            obj_dict[obj_name] = []

        # Only consider the trajectory after it has {rollout_per_task} rollouts
        if len(obj_dict[obj_name]) < rollout_per_task:
            rollout = load_json(file)
            
            if rollout[-1]["reward"] == 0.0:
                successful_rollouts.append(file)
                obj_dict[obj_name].append(file)

    # Verify that all objects have {rollout_per_task} rollouts
    for obj_name in obj_dict:
        if len(obj_dict[obj_name]) < rollout_per_task:
            print(f"Object {obj_name} has less than {rollout_per_task} rollouts. It has {len(obj_dict[obj_name])} rollouts")
    
    input("Press Enter to continue...")

    return successful_rollouts

def is_valid_rollout(rollout_path: str, rollout_per_task: int=-1) -> bool:
    """
    Returns:
        - True if the rollout is valid, False otherwise
    """
    is_a_rollout = rollout_path.endswith(".json") and "summary" not in rollout_path and "original" not in rollout_path
    
    if not is_a_rollout:
        return False
    
    if rollout_per_task == -1:
        return is_a_rollout
    
    rollout_idx = int(rollout_path.split("_")[-1].split(".")[0])
    return rollout_idx < rollout_per_task

def process_data(data_type: str, cfg: dict, i: int, mode: str):
    """
    Process the data for the given data type and iteration number

    Args:
        data_type (str): The type of data to process
        cfg (dict): The config for the dataproc
        i (int): The iteration number of the rollout
    """
    iter_str = f"iter{i}"

    if mode == "vanilla":
        raw_rollout_dir = os.path.join(cfg["logs_dir"], iter_str, data_type)  # input
    elif mode == "multi-star":
        rollout_iter_str = f"iter{i-1}"
        raw_rollout_dir = os.path.join(cfg["multi-star"][rollout_iter_str]["rollout_dir"], data_type)  # input
        iter_str += f"_multi-star{'_' + cfg['multi-star']['note'] if cfg['multi-star']['note'] != '' else ''}"
        iter_str += f"_mix-{int(100*cfg['multi-star']['pct_of_past_rollouts'])}pct-past" if cfg["multi-star"]["use_past_rollouts"] and i != 1 else ""
    elif mode == "leap":
        raise NotImplementedError("LEAP mode is not implemented for Guess My City")
        rollout_iter_str = f"iter{i-1}"
        raw_rollout_dir = os.path.join(cfg["leap"][rollout_iter_str]["rollout_dir"], data_type)  # input
        iter_str += "_leap"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    print(f"Raw rollout dir: {raw_rollout_dir}")

    data_dir = os.path.join(cfg["data_dir"], iter_str)  # output

    os.makedirs(data_dir, exist_ok=True)

    # Get path of all the json files in the raw_rollout_dir (ignore _summary_dict.json, which is only used to help generate rollouts)
    json_files = [os.path.join(raw_rollout_dir, f) for f in os.listdir(raw_rollout_dir) if is_valid_rollout(f, rollout_per_task=-1)]

    if mode == "multi-star":
        json_files = []

        if cfg["multi-star"]["use_past_rollouts"] and i != 1:
            # Past rollouts will be 50% of the total rollout per task
            past_rollout_per_task = int(cfg["multi-star"]["rollout_per_task"] * cfg["multi-star"]["pct_of_past_rollouts"]) 

            past_rollout_dirs = [os.path.join(cfg["multi-star"][f"iter{j}"][f"rollout_dir"], data_type) for j in range(0, i-1)]
            print(f"Past rollout dirs: {past_rollout_dirs}")
            past_rollout_per_task_per_dir = math.ceil(past_rollout_per_task / float(len(past_rollout_dirs)))
            for past_rollout_dir in past_rollout_dirs:
                print(f"Processing past rollout dir: {past_rollout_dir}")
                raw_past_rollout_files = [os.path.join(past_rollout_dir, f) for f in os.listdir(past_rollout_dir) if is_valid_rollout(f, rollout_per_task=past_rollout_per_task_per_dir)]
                json_files.extend(get_successful_rollouts(raw_past_rollout_files, past_rollout_per_task_per_dir))

            curr_rollout_per_task = cfg["multi-star"]["rollout_per_task"] - past_rollout_per_task
        else:
            curr_rollout_per_task = cfg["multi-star"]["rollout_per_task"]

        successful_json_files = [os.path.join(raw_rollout_dir, f) for f in os.listdir(raw_rollout_dir) if is_valid_rollout(f, rollout_per_task=cfg["multi-star"]["rollout_per_task"])]

        # Multi-Star only train on successful rollouts (so we need to filter out the failed rollouts)
        json_files.extend(get_successful_rollouts(successful_json_files, curr_rollout_per_task))
    elif mode == "leap":
        raise NotImplementedError("LEAP mode is not implemented for Guess My City")
        json_files = [f for f in json_files if int(f.split("_")[-1].split(".")[0]) in list(range(cfg["leap"]["rollout_per_task_range_min"], cfg["leap"]["rollout_per_task_range_max"]))]

    # Load the template
    with open(cfg["prompt_template_file"], "r") as f:
        prompt_template = Template(f.read())
    # And necessary parameters needed to render the template
    all_obj_list = [wv[0] for wv in get_default_city_list("all")]

    dataset = []

    if mode == "leap":
        num_successful_datapoints = 0
        num_relabeled_datapoints = 0 # failed rollouts with expert relabeled actions

    # Add each rollout to the dataset
    for file_path in tqdm(json_files, desc="Processing rollouts"):
        data = load_json(file_path)

        # Skip if the trajectory was not successful
        # if data[-1]["reward"] != 0.0:
        #     continue

        # Iterate over each timestep
        for i in range(len(data)):
            input_mode = "input" if i < 9 else "input_final"

            # Build the history
            observation_action_history = [
                {
                    "question": s["action"],
                    "answer": s["answer"]
                }
                for s in data[0:i]
            ]

            input_data = {
                "mode": input_mode,
                "all_city_list": all_obj_list,
                "observation_action_history": observation_action_history,
            }

            prompt = prompt_template.render(**input_data)

            if mode == "leap" and "expert_alternatives" in data[i]:
                # This is a failed rollout with expert relabeled actions
                output_data = {
                    "mode": "output",
                    "reason": data[i]["expert_alternatives"][0]["reason"],
                    "action": data[i]["expert_alternatives"][0]["action"]
                }
                num_relabeled_datapoints += 1
            else:
                output_data = {
                    "mode": "output",
                    "reason": data[i]["reason"],
                    "action": data[i]["action"]
                }

                if mode == "leap":
                    num_successful_datapoints += 1

            response = prompt_template.render(**output_data)

            datapoint = {
                "prompt": [{"role": "user", "content": prompt}],
                "response": [{"role": "assistant", "content": response}]
            }

            dataset.append(datapoint)

    print(f"Collected {len(dataset)} datapoints for {data_type}{f' (successful datapoints: {num_successful_datapoints}, relabeled datapoints: {num_relabeled_datapoints})' if mode == 'leap' else ''}")
    print(f"Saving the dataset to {os.path.join(data_dir, f'{data_type}.json')}")
    input("Press Enter to continue...")

    if mode == "multi-star" and "10k" in iter_str and data_type == "train":
        # We only keep the first 10k datapoints
        random.shuffle(dataset)
        dataset = dataset[:10000]
        print(f"Keeping only the first 10k datapoints for the training set")
        input("Press Enter to continue...")

    # Save the dataset
    save_json(os.path.join(data_dir, f"{data_type}.json"), dataset)

def main():
    args, cfg = preprocess_args()

    print(json.dumps(cfg, indent=4))
    input("Press Enter to continue...")

    for data_type in ["train", "val"]:
        process_data(data_type, cfg, args.i, args.mode)


if __name__ == "__main__":
    main()
