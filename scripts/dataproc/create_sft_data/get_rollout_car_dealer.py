"""
Typical usage:

If you are debugging (not using gpt4o), you can set the debug flag to True

python scripts/dataproc/create_sft_data/get_rollout_twenty_questions.py -t train -d
"""

import sys
import os
import argparse
import json
import yaml
import numpy as np
from typing import List, Dict
from jinja2 import Template

from agent_prm.envs.car_dealer.env import setup_car_dealer_env
from agent_prm.envs.car_dealer.data import TRAIN_BUYER_STRATEGIES, VAL_BUYER_STRATEGIES, TEST_BUYER_STRATEGIES, TRAIN_BRANDS, VAL_BRANDS, TEST_BRANDS, TRAIN_TYPES, VAL_TYPES, TEST_TYPES, TRAIN_FEATURES, VAL_FEATURES, TEST_FEATURES, DEFAULT_BRANDS, DEFAULT_TYPES, CAR_PRICES_BY_BRAND_AND_TYPE, CAR_FEATURES_ADDED_VALUE, format_car_options, format_chat_history, B2
from agent_prm.envs.car_dealer.interfaces import use_api

from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import load_json, save_json

HOST = "localhost"

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw car dealer logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/car_dealer.yaml", help='Path to car dealer dataproc config file')
    parser.add_argument('-t', '--data-types', nargs='+', default=["train", "val", "test"], help="A list of data types to process. Ex: train,val,test")
    parser.add_argument('-d', '--debug', default=False, action="store_true", help='Whether to run in debug mode (Human instead of gpt4o as the agent)')
    parser.add_argument('-e', '--activate-email', default=False, action="store_true", help='Whether to activate email logging')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-p', '--port', type=int, default=40042, help='Port number for the server')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def query_human(prev_api_call, prev_api_response):
    # Step 1: Use the API
    api_reason = "API reasoning placeholder"
    print(f"api_names: search_car_by_brand_type, search_car_by_brand, search_car_by_type, no_op")
    api_name = input("API name: ")
    if api_name != "no_op":
        api_brand = input("API brand (leave empty if none): ")
        api_type = input("API type (leave empty if none): ")
        # api_features = input("API features (leave empty if none): ")
        # api_features = [w.strip() for w in api_features.split(",")] if api_features else []

        api_call = {
            "api_name": api_name,
            "api_brand": api_brand,
            "api_type": api_type,
        }
        api_response = use_api(api_call)
    else:
        api_brand = ""
        api_type = ""

        api_call = prev_api_call
        api_response = prev_api_response

    # Step 2: Talk to the user based on the API response
    print(f"api_call: {api_call}")
    print(format_car_options(api_response))
    reason = "reason placeholder"
    action = input("Action: ")
    
    if api_response != {}:
        picked_car_idx = input("Picked car index: ")
        proposed_car = api_response[int(picked_car_idx)-1]
    else:
        proposed_car = {}

    print(f"proposed_car: {proposed_car}")

    cost = 0.0

    return api_reason, api_call, api_response, reason, action, proposed_car, cost

def query_expert(expert_agent_api_call_template: Template, expert_agent_template: Template, history: List[Dict[str, str]], prev_api_call: Dict[str, str], prev_api_response: Dict[str, str]):
    querying_cost = 0.0

    formated_history = format_chat_history(history)
    print(f"formated_history:\n{formated_history}\n")
    formatted_prev_api_response = format_car_options(prev_api_response)

    # Step 1: Get the system prompt
    system_prompt = expert_agent_api_call_template.render(system=True, all_car_brands=DEFAULT_BRANDS, all_car_types=DEFAULT_TYPES)
    input_prompt = expert_agent_api_call_template.render(system=False, mode="input", 
                                                         observation_action_history=formated_history,
                                                         previous_api_call=json.dumps(prev_api_call),
                                                         previous_api_response=formatted_prev_api_response
                                                         )
    
    # print(system_prompt)
    # print("--------------------------------")
    # print(input_prompt)
    # input("api call input prompt")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )
    print(f"gpt 4o cost (API call): {cost}")
    # print(response)
    # input("api call response")

    response_json = parse_json(response)
    try:
        assert response_json is not None, f"Failed to parse response: {response}"
        assert "reason" in response_json and "api_call" in response_json, f"Invalid response: {response_json}. Must contain 'reason' and 'api_call'"
    except Exception as e:
        elogger.log(f"Error parsing response: {response}")
        raise e
    
    querying_cost += cost

    api_reason = response_json["reason"]
    api_call = response_json["api_call"]
    api_response = use_api(api_call)

    if api_call["api_name"] == "no_op":
        api_call_used = prev_api_call
        api_response_used = prev_api_response
    else:
        api_call_used = api_call
        api_response_used = api_response

    # Step 2: Talk to the user based on the API response
    system_prompt = expert_agent_template.render(system=True, all_car_brands=DEFAULT_BRANDS, all_car_types=DEFAULT_TYPES)
    input_prompt = expert_agent_template.render(system=False, mode="input", 
                                                observation_action_history=formated_history,
                                                api_call=json.dumps(api_call_used),
                                                api_response=format_car_options(api_response_used)
                                                )
    # print(system_prompt)
    # print("--------------------------------")
    # print(input_prompt)
    # input("response input prompt")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )
    print(f"gpt 4o cost (response): {cost}")
    # print(f"api_call_used: {api_call_used}")
    # print(format_car_options(api_response_used))
    # print(response)
    # input("response")
    
    response_json = parse_json(response)
    try:
        assert response_json is not None, f"Failed to parse response: {response}"
        assert "reason" in response_json and "response" in response_json and "car_idx" in response_json, f"Invalid response: {response_json}. Must contain 'reason', 'response', and 'car_idx'"
    except Exception as e:
        elogger.log(f"Error parsing response: {response}")
        raise e

    querying_cost += cost

    # Determine the car index
    # print(f"api_response_used: {api_response_used}")
    if api_response_used != [] and int(response_json["car_idx"]) != 0:
        proposed_car = api_response_used[int(response_json["car_idx"])-1]
    else:
        proposed_car = {}

    # print(f"proposed_car: {proposed_car}")
    # input("proposed car")

    return api_reason, api_call, api_response, api_call_used, api_response_used, response_json["reason"], response_json["response"], proposed_car, querying_cost

def main():
    args, cfg = preprocess_args()
    np.random.seed(args.seed)
    elogger.set_activate(args.activate_email)
    env = setup_car_dealer_env(host=HOST, port=args.port)

    rollout_per_obj = cfg["rollout_per_obj"]

    with open(cfg["expert_api_call_template"], "r") as file:
        expert_agent_api_call_template = Template(file.read())
    with open(cfg["expert_response_template"], "r") as file:
        expert_agent_prompt_template = Template(file.read())
        
    with open("src/agent_prm/envs/car_dealer/car_inventory_dict.json", "r") as file:
        car_inventory_dict = json.load(file)

    all_games_to_play_list = []
    for data_type in args.data_types:
        if data_type == "train":
            buyer_strategy_list = TRAIN_BUYER_STRATEGIES
            brand_list = TRAIN_BRANDS
            type_list = TRAIN_TYPES
            feature_list = TRAIN_FEATURES
        elif data_type == "val":
            buyer_strategy_list = VAL_BUYER_STRATEGIES
            brand_list = VAL_BRANDS
            type_list = VAL_TYPES
            feature_list = VAL_FEATURES
        elif data_type == "test":
            buyer_strategy_list = TEST_BUYER_STRATEGIES
            brand_list = TEST_BRANDS
            type_list = TEST_TYPES
            feature_list = TEST_FEATURES

        data_type_all_games_to_play_list = []
        for buyer_strategy_idx in range(len(buyer_strategy_list)):
            for brand in brand_list:
                for car_type in type_list:
                    budget_list = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type]["budget"]
                    for budget in budget_list:
                        for rollout_idx in range(rollout_per_obj):
                            if buyer_strategy_list[buyer_strategy_idx] == B2:
                                # They will only buy if the car has all the features
                                #   car_inventory_dict[brand][car_type] gives us a list of in-stock cars
                                matching_car_idx = np.random.randint(1, len(car_inventory_dict[brand][car_type])) # Skip the first car because it's the base model
                                car_price = car_inventory_dict[brand][car_type][matching_car_idx]["msrp"]
                                features_to_include = car_inventory_dict[brand][car_type][matching_car_idx]["features"]
                            else:
                                features_to_include = list(np.random.choice(feature_list, size=np.random.randint(1, 4), replace=False))
                                car_price = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type]["msrp"] + sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in features_to_include])
                            
                            game_id = f"{buyer_strategy_idx}_{brand}_{car_type}_{budget}"
                            data_type_all_games_to_play_list.append((rollout_idx, game_id, data_type, buyer_strategy_list[buyer_strategy_idx], brand, car_type, budget, features_to_include, car_price))

        os.makedirs(os.path.join(cfg["logs_dir"], data_type), exist_ok=True)
        summary_dict_fp = os.path.join(cfg["logs_dir"], data_type, "_summary_dict.json")
        
        if not os.path.exists(summary_dict_fp):
            print(f"Summary dict not found at {summary_dict_fp}. Creating a new one.")
            summary_dict = {}
            save_json(summary_dict_fp, summary_dict)
        else:
            print(f"Loading summary dict from {summary_dict_fp}")
            summary_dict = load_json(summary_dict_fp)

        # Filter out games that have already been played
        data_type_all_games_to_play_list = [game for game in data_type_all_games_to_play_list if str(game[0]) not in summary_dict or game[1] not in summary_dict[str(game[0])]]
        all_games_to_play_list.extend(data_type_all_games_to_play_list)

    total_cost = 0.0

    for game in all_games_to_play_list:
        rollout_idx, game_id, data_type, buyer_strategy, brand, car_type, budget, features_to_include, car_price = game
        rollout_idx_str = str(rollout_idx)
        
        buyer_info = {
            "buyer_strategy": buyer_strategy,
            "preferred_brand": brand,
            "preferred_type": car_type,
            "preferred_features": features_to_include,
            "budget": budget,
            "msrp": car_price
        }
        print(f"Playing game {game_id}_{rollout_idx_str}")
        print(json.dumps(buyer_info, indent=4))
        
        history = env.reset()
        done = False
        total_reward = 0.0
        rollout_cost = 0.0
        traj_list = []
        
        prev_api_call = {}
        prev_api_response = {}

        while not done:
            if args.debug:
                api_reason, api_call, api_response, reason, action, proposed_car, cost = query_human(prev_api_call, prev_api_response)
                api_call_used = api_call
                api_response_used = api_response
                prev_api_call = api_call
                prev_api_response = api_response
            else:
                api_reason, api_call, api_response, api_call_used, api_response_used, reason, action, proposed_car, cost = query_expert(expert_agent_api_call_template, expert_agent_prompt_template, history, prev_api_call, prev_api_response)
                prev_api_call = api_call_used
                prev_api_response = api_response_used

            rollout_cost += cost

            print(f"++++++ agent step: {len(history)} ++++++")
            print(f"API Reason:\n{api_reason}\nAPI Call:\n{api_call}\nAPI Response:\n{format_car_options(api_response)}")
            print(f"Reason:\n{reason}\nAction:\n{action}")
            print(f"++++++ agent step: {len(history)}, total cost: {rollout_cost} ++++++")

            history, buyer_reason, buyer_response, buyer_decision, reward, success, failure_reason, done = env.step(buyer_info, history, action, proposed_car)
            total_reward += reward

            traj_list.append({
                "step": len(history)//2, # Because the history is doubled (buyer and seller)
                "api_reason": api_reason,
                "api_call": api_call,
                "api_response": api_response,
                "api_call_used": api_call_used,
                "api_response_used": api_response_used,
                "reason": reason,
                "action": action,
                "proposed_car": proposed_car,
                "buyer_reason": buyer_reason,
                "buyer_response": buyer_response,
                "buyer_decision": buyer_decision,
                "reward": reward,
                "success": success,
                "failure_reason": failure_reason,
                "cumulative_cost": rollout_cost,
            })

        total_cost += rollout_cost

        summary_dict_fp = os.path.join(cfg["logs_dir"], data_type, "_summary_dict.json")
        summary_dict = load_json(summary_dict_fp)

        if rollout_idx_str not in summary_dict:
            summary_dict[rollout_idx_str] = []
        summary_dict[rollout_idx_str].append(game_id)

        save_json(summary_dict_fp, summary_dict)
        save_json(os.path.join(cfg["logs_dir"], data_type, f"{game_id}_{rollout_idx_str}.json"), traj_list)

        print(f"======== collected idx={rollout_idx_str} game={game_id} with total reward {total_reward} and total cost {total_cost}")

    
if __name__ == "__main__":
    main()
    