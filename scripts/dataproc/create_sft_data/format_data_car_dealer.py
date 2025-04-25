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

Use case:
python format_data_car_dealer.py -i 0
"""

import json
import os
import argparse
import yaml
from jinja2 import Template

from agent_prm.envs.car_dealer.data import DEFAULT_BRANDS, DEFAULT_TYPES, format_car_options, format_chat_history

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw 20questions logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/car_dealer.yaml", help='Path to 20 questions dataproc config file')
    parser.add_argument('-i', type=int, required=True, help='The iteration number of the rollout')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def process_data(data_type: str, cfg: dict, i: int):
    """
    Process the data for the given data type and iteration number

    Args:
        data_type (str): The type of data to process
        cfg (dict): The config for the dataproc
        i (int): The iteration number of the rollout
    """
    if cfg["max_history_length"] is not None:
        iter_str = f"iter{i}-history-len-{cfg['max_history_length']}"
    else:
        iter_str = f"iter{i}"

    raw_rollout_dir = os.path.join(cfg["logs_dir"], data_type)  # input
    data_dir = os.path.join(cfg["data_dir"], iter_str)  # output

    os.makedirs(data_dir, exist_ok=True)

    # Get path of all the json files in the raw_rollout_dir (ignore _summary_dict.json, which is only used to help generate rollouts)
    json_files = [f for f in os.listdir(raw_rollout_dir) if f.endswith('.json') and not f.endswith('_summary_dict.json') and "original" not in f]

    # Load the template
    with open(cfg["api_call_template"], "r") as f:
        api_call_template = Template(f.read())
    with open(cfg["response_template"], "r") as f:
        response_template = Template(f.read())

    api_dataset = []
    response_dataset = []

    # Add each rollout to the dataset
    for file in json_files:
        file_path = os.path.join(raw_rollout_dir, file)

        # print(f"Processing {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        # Skip if the trajectory was not successful
        # if data[-1]["reward"] != 0.0:
        #     continue

        # Iterate over each timestep
        for i in range(len(data)):
            # Build the history
            observation_action_history = []
            for j in range(i):
                observation_action_history.append({
                    "role": "seller",
                    "content": data[j]["action"]
                })
                observation_action_history.append({
                    "role": "buyer",
                    "content": data[j]["buyer_response"]
                })

            if cfg["max_history_length"] is not None:
                observation_action_history = observation_action_history[-cfg["max_history_length"]:]

            observation_action_history = format_chat_history(observation_action_history)

            # print(f"Observation action history: {observation_action_history}")
            # input("====== observation action history ======")

            # 1. Generate the API call data
            prev_api_call = data[i-1]["api_call_used"] if i > 0 else {}
            prev_api_response = data[i-1]["api_response_used"] if i > 0 else []
            api_input_data = {
                "mode": 'input',
                "all_car_brands": DEFAULT_BRANDS,
                "all_car_types": DEFAULT_TYPES,
                "observation_action_history": observation_action_history,
                "previous_api_call": prev_api_call,
                "previous_api_response": format_car_options(prev_api_response)
            }
            api_prompt = api_call_template.render(**api_input_data)
            # print(api_prompt)
            # input("====== api prompt ======")

            api_output_data = {
                "mode": "output",
                "reason": data[i]["api_reason"],
                "api_name": data[i]["api_call"]["api_name"],
                "api_brand": "None" if data[i]["api_call"]["api_brand"] == "" else data[i]["api_call"]["api_brand"], # We ask the open source model to output None instead of an empty string
                "api_type": "None" if data[i]["api_call"]["api_type"] == "" else data[i]["api_call"]["api_type"]
            }
            api_response = api_call_template.render(**api_output_data)
            # print(api_response)
            # input("====== api response ======")

            # api_datapoint = {
            #     "prompt": [{"role": "user", "content": api_prompt}],
            #     "response": [{"role": "assistant", "content": api_response}]
            # }
            api_datapoint = {
                "prompt": api_prompt,
                "response": api_response
            }

            api_dataset.append(api_datapoint)

            # 2. Generate the response data
            response_input_data = {
                "mode": 'input',
                "all_car_brands": DEFAULT_BRANDS,
                "all_car_types": DEFAULT_TYPES,
                "observation_action_history": observation_action_history,
                "api_call": data[i]["api_call_used"],
                "api_response": format_car_options(data[i]["api_response_used"]),
            }
            response_prompt = response_template.render(**response_input_data)
            # print(response_prompt)
            # input("====== response prompt ======")

            # Unfortunately, we saved the proposed car instead of the index. We have to find it in the list of cars
            car_index = 0 # Default (when no car is proposed)
            proposed_car = data[i]["proposed_car"]
            if proposed_car != {}:
                for j in range(len(data[i]["api_response_used"])):
                    car = data[i]["api_response_used"][j]
                    if car["msrp"] == proposed_car["msrp"] and car["features"] == proposed_car["features"] and car["brand"] == proposed_car["brand"] and car["type"] == proposed_car["type"]:
                        car_index = j
                        break

            response_output_data = {
                "mode": "output",
                "reason": data[i]["reason"],
                "response": data[i]["action"],
                "car_idx": car_index
            }
            response_response = response_template.render(**response_output_data)
            # print(response_response)
            # input("====== response response ======")

            # response_datapoint = {
            #     "prompt": [{"role": "user", "content": response_prompt}],
            #     "response": [{"role": "assistant", "content": response_response}]
            # }
            response_datapoint = {
                "prompt": response_prompt,
                "response": response_response
            }
            response_dataset.append(response_datapoint)

    print(f"Collected {len(api_dataset)} API call datapoints and {len(response_dataset)} response datapoints for {data_type}")

    # Save the dataset
    with open(os.path.join(data_dir, f"api_call_{data_type}.json"), "w") as f:
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

    for data_type in ["train", "val", "test"]:
        process_data(data_type, cfg, args.i)


if __name__ == "__main__":
    main()
