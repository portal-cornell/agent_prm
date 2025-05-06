"""
Create training data for car dealer

We use the following template for the data

SFT format:
[
    {
        "prompt": "string: the prompt for the training data",
        "response": "string: the response for the training data"
    }
]
...

Usage:
- Vanilla mode (only training from the gpt4o rollouts):

python scripts/dataproc/create_sft_data/format_data_car_dealer.py -m vanilla -i 0
- Multi-Star mode (only training from the successful rollouts):

python scripts/dataproc/create_sft_data/format_data_car_dealer.py -m multi-star -i 1
- LEAP mode (training from successful rollouts + rollouts with expert relabeled actions):

python scripts/dataproc/create_sft_data/format_data_car_dealer.py -m leap -i 1
"""

import json
import os
import argparse
import yaml
import math
import random
from typing import List
from jinja2 import Template
from tqdm import tqdm

from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.envs.car_dealer.data import DEFAULT_BRANDS, DEFAULT_TYPES, DEFAULT_FEATURES, format_car_options, format_chat_history, format_api_call_history, format_most_recent_buyer_message

PAST_N = 3 # Number of previous api calls to include in the prompt

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw car dealer logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/car_dealer.yaml", help='Path to 20 questions dataproc config file')
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
    
    game_dict = {}

    for file in tqdm(json_files, desc="Processing rollouts"):
        buyer_type = file.split("/")[-1].split("_")[0]
        car_brand = file.split("/")[-1].split("_")[1]
        car_type = file.split("/")[-1].split("_")[2]

        game_id = f"{buyer_type}_{car_brand}_{car_type}"
        if game_id not in game_dict:
            game_dict[game_id] = []

        # Only consider the trajectory after it has {rollout_per_task} rollouts
        if len(game_dict[game_id]) < rollout_per_task:
            rollout = load_json(file)
            
            if rollout[-1]["success"]:
                successful_rollouts.append(file)
                game_dict[game_id].append(file)

    # Verify that all objects have {rollout_per_task} rollouts
    for game_id in game_dict:
        if len(game_dict[game_id]) < rollout_per_task:
            print(f"Game {game_id} has less than {rollout_per_task} rollouts. It has {len(game_dict[game_id])} rollouts")
    
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
    if cfg["max_history_length"] is not None:
        iter_str += f"-history-len-{cfg['max_history_length']}"

    if cfg["max_car_in_api_response"] is not None:
        iter_str += f"-max-car-{cfg['max_car_in_api_response']}"

    if mode == "vanilla":
        raw_rollout_dir = os.path.join(cfg["logs_dir"], data_type)  # input
    elif mode == "multi-star":
        rollout_iter_str = f"iter{i-1}"
        raw_rollout_dir = os.path.join(cfg["multi-star"][rollout_iter_str]["rollout_dir"], data_type)  # input
        iter_str += f"_multi-star{'_' + cfg['multi-star']['note'] if cfg['multi-star']['note'] != '' else ''}"
        iter_str += f"_mix-{int(100*cfg['multi-star']['pct_of_past_rollouts'])}pct-past" if cfg["multi-star"]["use_past_rollouts"] and i != 1 else ""
    elif mode == "leap":
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
                raw_past_rollout_files = [os.path.join(past_rollout_dir, f) for f in os.listdir(past_rollout_dir) if is_valid_rollout(f, rollout_per_task=-1)]
                json_files.extend(get_successful_rollouts(raw_past_rollout_files, past_rollout_per_task_per_dir))

            curr_rollout_per_task = cfg["multi-star"]["rollout_per_task"] - past_rollout_per_task
        else:
            curr_rollout_per_task = cfg["multi-star"]["rollout_per_task"]

        successful_json_files = [os.path.join(raw_rollout_dir, f) for f in os.listdir(raw_rollout_dir) if is_valid_rollout(f, rollout_per_task=-1)]

        # Multi-Star only train on successful rollouts (so we need to filter out the failed rollouts)
        json_files.extend(get_successful_rollouts(successful_json_files, curr_rollout_per_task))
    elif mode == "leap":
        json_files = [f for f in json_files if int(f.split("_")[-1].split(".")[0]) in list(range(cfg["leap"]["rollout_per_task_range_min"], cfg["leap"]["rollout_per_task_range_max"]))]

    # Load the template
    with open(cfg["api_call_template"], "r") as f:
        api_call_template = Template(f.read())
    with open(cfg["response_template"], "r") as f:
        response_template = Template(f.read())

    api_dataset = []
    response_dataset = []

    # Add each rollout to the dataset
    for file in tqdm(json_files, desc="Processing rollouts"):
        file_path = file

        # print(f"Processing {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Skip if the trajectory was not successful
        # if data[-1]["reward"] != 0.0:
        #     continue

        # Iterate over each timestep
        for i in range(len(data)):
            # Build the history
            history = []
            all_api_calls = [] # List[Dict]
            all_api_calls_have_responses = [] # List[bool]
            for j in range(i):
                history.append({
                    "role": "seller",
                    "content": data[j]["action"]
                })
                history.append({
                    "role": "buyer",
                    "content": data[j]["buyer_response"]
                })

                all_api_calls.append(data[j]["api_call"])
                all_api_calls_have_responses.append(data[j]["api_response"] != [])
            
            observation_action_history = history
            if cfg["max_history_length"] is not None:
                observation_action_history = history[-cfg["max_history_length"]:]

            observation_action_history = format_chat_history(observation_action_history)

            # print(f"Observation action history: {observation_action_history}")
            # input("====== observation action history ======")

            # 1. Generate the API call data
            prev_api_call = data[i-1]["api_call_used"] if i > 0 else {}
            if i > 0:
                if "car_list" in data[i-1]:
                    # If the rollouts used max car, we need to use the car list because there's stochasticity in the api response
                    prev_api_response = data[i-1]["car_list"]
                    prev_api_response_str, _ = format_car_options(prev_api_response, max_car=-1)
                else:
                    prev_api_response = data[i-1]["api_response_used"]
                    prev_api_response_str, _ = format_car_options(prev_api_response, cfg["max_car_in_api_response"])
            else:
                prev_api_response = []
            api_input_data = {
                "mode": 'input',
                "all_car_brands": DEFAULT_BRANDS,
                "all_car_types": DEFAULT_TYPES,
                "all_car_features": DEFAULT_FEATURES,
                "observation_action_history": observation_action_history,
                "past_N": PAST_N,
                "prev_api_call_history": format_api_call_history(all_api_calls, all_api_calls_have_responses, PAST_N),
                "previous_api_call": prev_api_call,
                "previous_api_response": prev_api_response_str
            }
            api_prompt = api_call_template.render(**api_input_data).strip()

            api_output_data = {
                "mode": "output",
                "reason": data[i]["api_reason"],
                "api_name": data[i]["api_call"]["api_name"],
                "api_brand": "None" if data[i]["api_call"]["api_brand"] == "" else data[i]["api_call"]["api_brand"], # We ask the open source model to output None instead of an empty string
                "api_type": "None" if data[i]["api_call"]["api_type"] == "" else data[i]["api_call"]["api_type"],
                "api_features": data[i]["api_call"]["api_features"]
            }
            api_response = api_call_template.render(**api_output_data).strip()
            # print(api_response)
            # input("====== api response ======")

            api_datapoint = {
                "prompt": [{"role": "user", "content": api_prompt}],
                "response": [{"role": "assistant", "content": api_response}]
            }
            # api_datapoint = {
            #     "prompt": api_prompt,
            #     "response": api_response
            # }

            api_dataset.append(api_datapoint)

            # 2. Generate the response data

            if "car_list" in data[i-1]:
                api_response_used = data[i-1]["car_list"]
                # Since this has already used the max car, we don't want to further truncate the car list
                api_response_str, _ = format_car_options(api_response_used, max_car=-1) 
            else:
                api_response_used = data[i-1]["api_response_used"]
                api_response_str, _ = format_car_options(api_response_used, cfg["max_car_in_api_response"])
            response_input_data = {
                "mode": 'input',
                "all_car_brands": DEFAULT_BRANDS,
                "all_car_types": DEFAULT_TYPES,
                "observation_action_history": observation_action_history,
                "api_call": data[i]["api_call_used"],
                "api_response": api_response_str,
                "buyer_response": format_most_recent_buyer_message(history)
            }
            response_prompt = response_template.render(**response_input_data).strip()

            # Unfortunately, we saved the proposed car instead of the index. We have to find it in the list of cars
            car_index = 0 # Default (when no car is proposed)
            proposed_car = data[i]["proposed_car"]
            if proposed_car != {}:
                for j in range(len(data[i]["api_response_used"])):
                    car = data[i]["api_response_used"][j]
                    if car["msrp"] == proposed_car["msrp"] and car["features"] == proposed_car["features"] and car["brand"] == proposed_car["brand"] and car["type"] == proposed_car["type"]:
                        car_index = j + 1 # +1 because we printed out the api response from 1
                        break

            response_output_data = {
                "mode": "output",
                "reason": data[i]["reason"],
                "response": data[i]["action"],
                "car_idx": car_index,
                # We want to make the model learn to copy down the proposed car brand, type, features, and msrp
                "proposed_car_brand": proposed_car["brand"] if proposed_car != {} else "None",
                "proposed_car_type": proposed_car["type"] if proposed_car != {} else "None",
                "proposed_car_features": proposed_car["features"] if proposed_car != {} else [],
                "proposed_car_msrp": proposed_car["msrp"] if proposed_car != {} else 0
            }
            response_response = response_template.render(**response_output_data).strip()
            # print(response_response)
            # input("====== response response ======")

            response_datapoint = {
                "prompt": [{"role": "user", "content": response_prompt}],
                "response": [{"role": "assistant", "content": response_response}]
            }

            # response_datapoint = {
            #     "prompt": response_prompt,
            #     "response": response_response
            # }
            response_dataset.append(response_datapoint)

    print(f"Collected {len(api_dataset)} API call datapoints and {len(response_dataset)} response datapoints for {data_type}")
    print(f"Saving the dataset to {os.path.join(data_dir, f'{data_type}.json')}")
    input("Press Enter to continue...")

    if mode == "multi-star" and "10k" in iter_str and data_type == "train":
        # We only keep the first 10k datapoints
        random.shuffle(response_dataset)
        response_dataset = response_dataset[:10000]
        random.shuffle(api_dataset)
        api_dataset = api_dataset[:10000]
        print(f"Keeping only the first 10k datapoints for the training set")
        input("Press Enter to continue...")

    # Save the dataset
    with open(os.path.join(data_dir, f"api_{data_type}.json"), "w") as f:
        json.dump(api_dataset, f, indent=4)
    with open(os.path.join(data_dir, f"response_{data_type}.json"), "w") as f:
        json.dump(response_dataset, f, indent=4)

    # Merge the API call and response datasets
    merged_dataset = api_dataset + response_dataset
    print(f"Merged {len(merged_dataset)} datapoints for {data_type}")
    with open(os.path.join(data_dir, f"{data_type}.json"), "w") as f:
        json.dump(merged_dataset, f, indent=4)

def main():
    args, cfg = preprocess_args()

    print(json.dumps(cfg, indent=4))
    input("Check config")

    for data_type in ["train", "val"]:
        process_data(data_type, cfg, args.i, args.mode)


if __name__ == "__main__":
    main()
