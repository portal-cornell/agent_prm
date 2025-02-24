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
...
"""

import json
import os
import argparse
import yaml
from jinja2 import Template
from agent_prm.envs.twenty_questions.data import get_default_word_list

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw 20questions logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/20questions.yaml", help='Path to 20 questions dataproc config file')
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
    iter_str = f"iter{i}"

    raw_rollout_dir = os.path.join(cfg["logs_dir"], iter_str, data_type)  # input
    data_dir = os.path.join(cfg["data_dir"], iter_str)  # output

    os.makedirs(data_dir, exist_ok=True)

    # Get path of all the json files in the raw_rollout_dir (ignore _summary_dict.json, which is only used to help generate rollouts)
    json_files = [f for f in os.listdir(raw_rollout_dir) if f.endswith('.json') and not f.endswith('_summary_dict.json')]

    # Load the template
    with open(cfg["prompt_template_file"], "r") as f:
        prompt_template = Template(f.read())
    # And necessary parameters needed to render the template
    all_obj_list = [wv[0] for wv in get_default_word_list("all")]

    dataset = []

    # Add each rollout to the dataset
    for file in json_files:
        file_path = os.path.join(raw_rollout_dir, file)

        with open(file_path, "r") as f:
            data = json.load(f)

        # Skip if the trajectory was not successful
        if data[-1]["reward"] != 0.0:
            continue

        # Iterate over each timestep
        for i in range(len(data)):
            input_mode = "input" if i < 19 else "input_final"

            # Build the history
            observation_action_history = [
                {
                    "question": s["question"],
                    "answer": s["answer"]
                }
                for s in data[0:i]
            ]

            input_data = {
                "mode": input_mode,
                "all_obj_list": all_obj_list,
                "observation_action_history": observation_action_history,
            }

            prompt = prompt_template.render(**input_data)

            output_data = {
                "mode": "output",
                "reason": data[i]["reason"],
                "action": data[i]["question"]
            }

            response = prompt_template.render(**output_data)

            datapoint = {
                "prompt": [{"role": "user", "content": prompt}],
                "response": [{"role": "assistant", "content": response}]
            }

            dataset.append(datapoint)

    print(f"Collected {len(dataset)} datapoints for {data_type}")

    # Save the dataset
    with open(os.path.join(data_dir, f"{data_type}.json"), "w") as f:
        json.dump(dataset, f, indent=4)

def main():
    args, cfg = preprocess_args()

    for data_type in ["train", "val", "test"]:
        process_data(data_type, cfg, args.i)


if __name__ == "__main__":
    main()
