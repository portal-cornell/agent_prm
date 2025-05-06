"""
Typical usage:

If you are debugging (not using gpt4o), you can set the debug flag to True

python scripts/dataproc/create_sft_data/get_rollout_car_dealer.py -t train -sh TODO -sp TODO
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
from agent_prm.envs.car_dealer.data import DEFAULT_BRANDS, DEFAULT_TYPES, DEFAULT_FEATURES
from agent_prm.envs.car_dealer.data import format_car_options, format_chat_history, load_car_inventories, determine_car_inventory, get_all_games_to_play, format_api_call_history
from agent_prm.envs.car_dealer.interface import use_api

from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import load_json, save_json

def preprocess_args():
    parser = argparse.ArgumentParser(description='Generate raw car dealer logs')
    parser.add_argument('--config', type=str, default="configs/create_sft_training_data/car_dealer.yaml", help='Path to car dealer dataproc config file')
    parser.add_argument('-t', '--data-types', nargs='+', default=["train", "val", "test"], help="A list of data types to process. Ex: train,val,test")
    parser.add_argument('-d', '--debug', default=False, action="store_true", help='Whether to run in debug mode (Human instead of gpt4o as the agent)')
    parser.add_argument('-e', '--activate-email', default=False, action="store_true", help='Whether to activate email logging')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-sh', '--sim_host', type=str, default="localhost", help='Host name for the simulator')
    parser.add_argument('-sp', '--sim_port', type=int, default=40042, help='Port number for the simulator')
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
    print(format_car_options(api_response)[0])
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

MAX_QUERY_ATTEMPTS = 3
def query_expert(expert_agent_api_call_template: Template, 
                 expert_agent_template: Template, 
                 history: List[Dict[str, str]], 
                 prev_api_call: Dict[str, str], 
                 prev_api_response: Dict[str, str], 
                 all_prev_api_calls: List[Dict[str, str]], 
                 all_prev_api_calls_have_responses: List[bool], 
                 buyer_info: dict, 
                 car_inventories: dict,
                 past_N: int = 3):
    querying_cost = 0.0

    formated_history = format_chat_history(history)
    formatted_prev_api_response = format_car_options(prev_api_response)[0]

    # Step 1: Get the system prompt
    system_prompt = expert_agent_api_call_template.render(system=True, all_car_brands=DEFAULT_BRANDS, all_car_types=DEFAULT_TYPES, all_car_features=DEFAULT_FEATURES).strip()
    input_prompt = expert_agent_api_call_template.render(system=False, mode="input", 
                                                         observation_action_history=formated_history,
                                                         past_N=past_N,
                                                         prev_api_call_history=format_api_call_history(all_prev_api_calls, all_prev_api_calls_have_responses, past_N),
                                                         previous_api_call=json.dumps(prev_api_call),
                                                         previous_api_response=formatted_prev_api_response
                                                         ).strip()
    
    # print(system_prompt)
    # print("--------------------------------")
    # print(input_prompt)
    # input("api call input prompt")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    terminate = False
    query_attempts = 0

    while not terminate and query_attempts < MAX_QUERY_ATTEMPTS:
        response, cost = generate_from_openai_completion(
            messages=messages, model="gpt-4o"
        )
        # print(f"gpt 4o cost (API call): {cost}")
        # print(response)
        # input("api call response")

        response_json = parse_json(response)
        try:
            assert response_json is not None, f"Failed to parse response: {response}"
            assert "reason" in response_json and "api_call" in response_json, f"Invalid response: {response_json}. Must contain 'reason' and 'api_call'"
            terminate = True
        except Exception as e:
            print(f"Error parsing API call response: {response}")
            # elogger.log(f"Error parsing response: {response}")

        query_attempts += 1
        querying_cost += cost

    if not terminate:
        elogger.log(f"Failed to get a valid response after {MAX_QUERY_ATTEMPTS} attempts")
        raise Exception(f"Failed to get a valid response after {MAX_QUERY_ATTEMPTS} attempts")

    # Determine the car inventory to use
    car_inventory_dict = determine_car_inventory(buyer_info, car_inventories)

    api_reason = response_json["reason"]
    api_call = response_json["api_call"]
    api_response = use_api(api_call, car_inventory_dict)

    if api_call["api_name"] == "no_op":
        api_call_used = prev_api_call
        api_response_used = prev_api_response
    else:
        api_call_used = api_call
        api_response_used = api_response

    # Step 2: Talk to the user based on the API response
    system_prompt = expert_agent_template.render(system=True, all_car_brands=DEFAULT_BRANDS, all_car_types=DEFAULT_TYPES).strip()
    input_prompt = expert_agent_template.render(system=False, mode="input", 
                                                observation_action_history=formated_history,
                                                api_call=json.dumps(api_call_used),
                                                api_response=format_car_options(api_response_used)
                                                ).strip()
    # print(system_prompt)
    # print("--------------------------------")
    # print(input_prompt)
    # input("response input prompt")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    terminate = False
    query_attempts = 0

    while not terminate and query_attempts < MAX_QUERY_ATTEMPTS:
        response, cost = generate_from_openai_completion(
            messages=messages, model="gpt-4o"
        )
    
        response_json = parse_json(response)
        try:
            assert response_json is not None, f"Failed to parse response: {response}"
            assert "reason" in response_json and "response" in response_json and "car_idx" in response_json and "proposed_car" in response_json, f"Invalid response: {response_json}. Must contain 'reason', 'response', 'car_idx', and 'proposed_car'"
            assert "brand" in response_json["proposed_car"] and "type" in response_json["proposed_car"] and "features" in response_json["proposed_car"] and "msrp" in response_json["proposed_car"], f"Invalid proposed car: {response_json['proposed_car']}"
            terminate = True
        except Exception as e:
            print(f"Error parsing response: {response}")
            # elogger.log(f"Error parsing response: {response}")

        query_attempts += 1
        querying_cost += cost

    if not terminate:
        elogger.log(f"Failed to get a valid response after {MAX_QUERY_ATTEMPTS} attempts")
        raise Exception(f"Failed to get a valid response after {MAX_QUERY_ATTEMPTS} attempts")

    # Determine the car index
    # print(f"api_response_used: {api_response_used}")
    if api_response_used != [] and int(response_json["car_idx"]) != 0:
        proposed_car = api_response_used[int(response_json["car_idx"])-1]
    else:
        proposed_car = {}

    # print(f"proposed_car: {proposed_car}")
    # input("proposed car")

    proposed_car_copied_in_response = response_json["proposed_car"]

    return api_reason, api_call, api_response, api_call_used, api_response_used, response_json["reason"].strip(), response_json["response"].strip(), proposed_car, proposed_car_copied_in_response, querying_cost

def main():
    args, cfg = preprocess_args()
    np.random.seed(args.seed)
    elogger.set_activate(args.activate_email)
    env = setup_car_dealer_env(host=args.sim_host, port=args.sim_port)

    rollout_per_obj = cfg["rollout_per_obj"]

    with open(cfg["expert_api_call_template"], "r") as file:
        expert_agent_api_call_template = Template(file.read())
    with open(cfg["expert_response_template"], "r") as file:
        expert_agent_prompt_template = Template(file.read())
        
    car_inventories = load_car_inventories()

    # A list of tuples
    #  (rollout_idx, game_id, data_type, buyer_info)
    all_games_to_play_list = get_all_games_to_play(args.data_types, cfg["logs_dir"], range(rollout_per_obj))

    print(all_games_to_play_list)
    print(f"num of games to play: {len(all_games_to_play_list)}")
    input("all_games_to_play_list")

    total_cost = 0.0

    for game in all_games_to_play_list:
        rollout_idx, game_id, data_type, buyer_info = game
        rollout_idx_str = str(rollout_idx)

        print(f"Playing game {game_id}_{rollout_idx_str}")
        print(json.dumps(buyer_info, indent=4))
        
        history = env.reset()
        done = False
        total_reward = 0.0
        rollout_cost = 0.0
        traj_list = []
        
        all_prev_api_calls = [] # List[Dict]
        all_prev_api_calls_have_responses = [] # List[bool] whether able to find any car
        prev_api_call = {}
        prev_api_response = {}
        num_negotiation = 0
        num_car_proposed = 0
        prev_proposed_car = {}

        while not done:
            if args.debug:
                assert False, "Deprecated. Have not updated since 4/24 change"
                api_reason, api_call, api_response, reason, action, proposed_car, cost = query_human(prev_api_call, prev_api_response)
                api_call_used = api_call
                api_response_used = api_response
                prev_api_call = api_call
                prev_api_response = api_response
            else:
                api_reason, api_call, api_response, api_call_used, api_response_used, reason, action, proposed_car, proposed_car_copied_in_response, cost = query_expert(expert_agent_api_call_template, expert_agent_prompt_template, history, prev_api_call, prev_api_response, all_prev_api_calls, all_prev_api_calls_have_responses, buyer_info, car_inventories)
                prev_api_call = api_call_used
                prev_api_response = api_response_used

            rollout_cost += cost

            all_prev_api_calls.append(api_call)
            all_prev_api_calls_have_responses.append(api_response != [])

            print(f"++++++ agent step: {len(history)//2} ++++++")
            print(f"API Reason:\n{api_reason}\nAPI Call:\n{api_call}\nAPI Response:\n{format_car_options(api_response)[0]}")
            print(f"Reason:\n{reason}\nAction:\n{action}\nProposed Car:\n{proposed_car}\nProposed Car Copied in Response:\n{proposed_car_copied_in_response}")
            print(f"++++++ agent step: {len(history)//2}, total cost: {rollout_cost} ++++++")

            history, buyer_reason, buyer_response, buyer_decision, reward, success, failure_reason, done, num_negotiation, proposed_car, num_car_proposed = env.step(buyer_info, history, action, proposed_car, proposed_car_copied_in_response, num_negotiation, prev_proposed_car, num_car_proposed, car_inventories)
            prev_proposed_car = proposed_car
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
                "num_negotiation": num_negotiation,
                "num_car_proposed": num_car_proposed,
                "cumulative_cost": rollout_cost,
            })

            if len(traj_list) == 1:
                traj_list[0]["buyer_info"] = buyer_info  # Also add the buyer info to the first step

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
    