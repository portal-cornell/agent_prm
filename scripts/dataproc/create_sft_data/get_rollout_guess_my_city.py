"""
Typical usage:

If you are debugging (not using gpt4o), you can set the debug flag to True

python scripts/dataproc/create_sft_data/get_rollout_guess_my_city.py -t train -d
"""

import sys
import os
import argparse
import json
import yaml
from typing import List, Dict
from jinja2 import Template

from agent_prm.envs.guess_my_city.env import setup_guess_my_city_env
from agent_prm.envs.guess_my_city.data import TRAIN_CITY_DICT, VALIDATION_CITY_DICT, TEST_CITY_DICT, WordVariants, get_default_word_list 
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import load_json, save_json

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw guess_my_city logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/guess_my_city.yaml", help='Path to guess my city dataproc config file')
    parser.add_argument('-t', '--data-type', type=str, required=True, choices=["train", "val", "test"], help='Whether to use the train, or validation, or test set')
    parser.add_argument('-d', '--debug', default=False, action="store_true", help='Whether to run in debug mode (Human instead of gpt4o as the agent)')
    parser.add_argument('-e', '--activate-email', default=False, action="store_true", help='Whether to activate email logging')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def query_expert(history: List[Dict[str, str]], expert_agent_prompt_template: Template, all_obj_list: List[WordVariants], last_question: bool = False):
    system_prompt = expert_agent_prompt_template.render(system=True, all_obj_list=all_obj_list)
    input_prompt = expert_agent_prompt_template.render(system=False, mode="input" if not last_question else "input_final", observation_action_history=history)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )
    print(f"gpt 4o cost: {cost}")

    response_json = parse_json(response)
    try:
        assert response_json is not None, f"Failed to parse response: {response}"
        assert "reason" in response_json and "question" in response_json, f"Invalid response: {response_json}. Must contain 'reason' and 'question'"
    except Exception as e:
        elogger.log(f"Error parsing response: {response}")
        raise e

    return response_json["reason"], response_json["question"], cost


def main():
    args, cfg = preprocess_args()
    elogger.set_activate(args.activate_email)
    env = setup_guess_my_city_env()
    all_city_list = [wv[0] for wv in get_default_word_list("all")]

    rollout_per_city = cfg["rollout_per_obj"]

    if args.data_type == "train":
        city_dict_to_use = TRAIN_CITY_DICT 
    elif args.data_type == "val":
        city_dict_to_use = VALIDATION_CITY_DICT 
    elif args.data_type == "test":
        city_dict_to_use = TEST_CITY_DICT 
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    with open(cfg["expert_template"], "r") as file:
        expert_agent_prompt_template = Template(file.read())

    os.makedirs(os.path.join(cfg["logs_dir"], args.data_type), exist_ok=True)
    summary_dict_fp = os.path.join(cfg["logs_dir"], args.data_type, "_summary_dict.json")
    
    if not os.path.exists(summary_dict_fp):
        print(f"Summary dict not found at {summary_dict_fp}. Creating a new one.")
        summary_dict = {}
        save_json(summary_dict_fp, summary_dict)
    else:
        print(f"Loading summary dict from {summary_dict_fp}")
        summary_dict = load_json(summary_dict_fp)

    total_cost = 0.0

    for rollout_idx in range(rollout_per_city):
        rollout_idx_str = str(rollout_idx)
        if rollout_idx_str not in summary_dict:
            summary_dict[rollout_idx_str] = []

        for category in city_dict_to_use.keys():
            for city in city_dict_to_use[category]:
                if city in summary_dict[rollout_idx_str]:
                    print(f"Skipping {city} as it is already in the summary dict")
                    continue
                city_to_process = WordVariants.from_str(city)
                
                history = env.reset(city=city_to_process)
                done = False
                total_reward = 0.0
                rollout_cost = 0.0
                traj_list = []

                while not done:
                    last_question = len(history) == env.max_conversation_length - 1

                    if args.debug:
                        reason = input("Reason: ")
                        action = input("Action: ")
                        cost = 0.0
                    else:
                        reason, action, cost = query_expert(history, expert_agent_prompt_template, all_city_list, last_question)

                    rollout_cost += cost

                    print(f"++++++ agent step: {len(history)} ++++++")
                    print(f"Reason:\n{reason}\nAction:\n{action}", )
                    print(f"++++++ agent step: {len(history)}, total cost: {rollout_cost} ++++++")

                    obs, reward, done = env.step(history, action)
                    history, answerer_reason, answer = obs
                    total_reward += reward

                    traj_list.append({
                        "reason": reason,
                        "action": action,
                        "answerer_reason": answerer_reason,
                        "answer": answer,
                        "reward": reward,
                        "cumulative_cost": rollout_cost,
                    })

                total_cost += rollout_cost

                summary_dict[rollout_idx_str].append(city)

                save_json(summary_dict_fp, summary_dict)
                save_json(os.path.join(cfg["logs_dir"], args.data_type, f"{city}_{rollout_idx_str}.json"), traj_list)

                print(f"======== collected idx={rollout_idx_str} obj={city} with total reward {total_reward} and total cost {total_cost}")

    
if __name__ == "__main__":
    main()
    