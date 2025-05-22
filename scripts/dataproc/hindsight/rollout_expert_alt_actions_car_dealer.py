"""
Example usage:

When getting export's alternative actions:
    python scripts/dataproc/hindsight/rollout_expert_alt_actions_car_dealer.py -e -m g -i 0 -d train -min 0 -max 4

    where
        -e indicates that we are using elogger
        -m g indicates that we are generating rollouts with alt actions
        -i 1 indicates the iteration that we are on (which affects the rollout directory)
        -d val indicates that we are processing the validation set.

When completing the rollouts:
    Without specifying the object range:
        python scripts/dataproc/hindsight/rollout_expert_alt_actions_car_dealer.py -e -m r -i 0 -d train -min 4 -max 20 --sim_host TODO --sim_port TODO

        where
            -e indicates that we are using elogger
            -m r indicates that we are completing the rollouts
            -i 1 indicates the iteration that we are on (which affects the rollout directory and the agent config)
            -d train indicates that we are processing the training set
            --min 4 --max 12 indicates that we are completing the rollouts from idx 4 to 12
            -s indicates that we are serving the model

    python scripts/dataproc/hindsight/rollout_expert_alt_actions_car_dealer.py -e -m r -i 1 -d train -min 4 -max 12 -si 0 -ei 27 -et gpt4o --sim_host TODO --sim_port TODO
        where
            -si 0 -ei 27 indicates that we are starting from object 0 and ending at object 27 (not inclusive)

When merging the summary dicts:
    python scripts/dataproc/hindsight/rollout_expert_alt_actions_car_dealer.py -m merge_gen -i 0 -d train

    python scripts/dataproc/hindsight/rollout_expert_alt_actions_car_dealer.py -m merge_rollout -i 0 -d val
"""
import argparse
import os
import random
import json
import math
import time
import copy
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Tuple
from jinja2 import Template

from agent_prm.envs.car_dealer.env import BatchedCarDealerEnvironment, setup_batched_car_dealer_env, are_same_cars
from agent_prm.envs.car_dealer.data import TRAIN_BUYER_STRATEGIES, VAL_BUYER_STRATEGIES, TEST_BUYER_STRATEGIES, TRAIN_BRANDS, VAL_BRANDS, TEST_BRANDS, TRAIN_TYPES, VAL_TYPES, TEST_TYPES, DEFAULT_BRANDS, DEFAULT_TYPES, DEFAULT_FEATURES, format_chat_history, format_car_options, format_api_call_history, determine_car_inventory, load_car_inventories, check_has_discount
from agent_prm.envs.car_dealer.parser import parse_reason_and_action_car_dealer
from agent_prm.envs.car_dealer.interface import use_api, rollout_batch

from agent_prm.agents.agent_registry import initialize_agent
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import start_sglang_server
from agent_prm.agents.agent import Agent


iter_to_rollout_dir = {
    # pi0 3 epochs
    0: "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter0/pi0-62pct_max-car-8_250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3", 
    # pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50
    1: "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter1/pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50_250506_185257_iter1_hindsight-biased-on-50_pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50",
    # pi2-80pct_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50
    2: "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter2/pi2-80pct_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50_250510_115714_iter2_hindsight-biased-on-50_no-past-rollout_pi2_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50"
}


iter_to_agent_config = {
    0: {
        "type": "sglang_server",
        "log_name": "pi0-62pct_max-car-8",
        "model_id": "/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93",
        "api_prompt_template_file": "prompts/car_dealer/car_dealer_api_template.j2",
        "prompt_template_file": "prompts/car_dealer/car_dealer_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    },
    1: {
        "type": "sglang_server",
        "log_name": "pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50",
        "model_id": "/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250506_185257_iter1_hindsight-biased-on-50_pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50",
        "api_prompt_template_file": "prompts/car_dealer/car_dealer_api_template.j2",
        "prompt_template_file": "prompts/car_dealer/car_dealer_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    },
    2: {
        "type": "sglang_server",
        "log_name": "pi2-80pct_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50",
        "model_id": "/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250510_115714_iter2_hindsight-biased-on-50_no-past-rollout_pi2_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50/checkpoint-800",
        "api_prompt_template_file": "prompts/car_dealer/car_dealer_api_template.j2",
        "prompt_template_file": "prompts/car_dealer/car_dealer_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    }
}

NUM_ALT_RESPONSES = 0

MAX_QUERY_ATTEMPTS = 3
with open("prompts/car_dealer/car_dealer_summary_history_template.j2", "r") as f:
    car_dealer_summary_history_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_summary.j2", "r") as f:
    car_dealer_summary_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_api_expert_gen_prefered_api_action.j2", "r") as f:
    car_dealer_api_expert_gen_prefered_api_action_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_api_expert_gen_prefered_response_action.j2", "r") as f:
    car_dealer_api_expert_gen_prefered_response_action_template = Template(f.read())

def get_game_id_and_rollout_idx_from_file_name(f: str) -> Tuple[str, int]:
    # The last element is the rollout index
    game_id_element_list = f.split("_")[:-1]
    game_id = "_".join(game_id_element_list)
    rollout_idx = int(f.split("_")[-1].split(".")[0])
    return game_id, rollout_idx

def is_valid_rollout(f: str) -> bool:
    """
    Check if the rollout is valid
    """
    return f.endswith(".json") and not f.endswith("_original.json") and not "summary_dict" in f

def is_within_valid_range(file_name: str, rollout_idx_min: int, rollout_idx_max: int) -> bool:
    """
    Check if the rollout idx is within the valid range
    """
    if rollout_idx_min == -1 and rollout_idx_max == -1:
        return True

    rollout_idx = int(file_name.split("_")[-1].split(".")[0])

    return rollout_idx >= rollout_idx_min and rollout_idx < rollout_idx_max

def need_to_complete(file_name: str, summary_dict: Dict) -> bool:
    """
    Check if the rollout needs to be completed
    """
    game_id, rollout_idx = get_game_id_and_rollout_idx_from_file_name(file_name)

    return str(rollout_idx) not in summary_dict or game_id not in summary_dict[str(rollout_idx)]

def format_detailed_history(rollout: List[Dict]):
    history_str = ""
    for i in range(1, len(rollout)):
        step = rollout[i]

        _, action = parse_reason_and_action_car_dealer(step["raw_text"])

        # sort the key of car_suggested
        car_suggested = step["proposed_car"]
        car_suggested = {k: car_suggested[k] for k in sorted(car_suggested.keys())}

        copied_car_info = action["proposed_car"]
        copied_car_info = {k: copied_car_info[k] for k in sorted(copied_car_info.keys())}

        step_str = car_dealer_summary_history_template.render(
            step_idx=i,
            api_attempted=step["api_call"],
            api_response=format_car_options(step["api_response"])[0],
            api_used = step["api_call_used"],
            api_response_used = format_car_options(step["api_response_used"])[0],
            message_to_buyer = step["action"],
            car_suggested = car_suggested,
            copied_car_info = copied_car_info,
            buyer_response = step["buyer_response"],
        ).strip()

        history_str += step_str + "\n\n"

    return history_str

def format_failure_reason(rollout: List[Dict]):
    if rollout[-1]["success"]:
        return "The seller successfully sold the car to the buyer."
    
    failure_reason = rollout[-1]["failure_reason"]

    if len(rollout) == 10:
        # Maximum number of steps
        failure_reason += " The seller failed to convince the buyer to buy the car within 10 steps."

    return failure_reason

def gen_summary_from_rollout(rollout: List[Dict]):
    """
    Generate a summary from the rollout
    """
    history_str = format_detailed_history(rollout).strip()
    failure_reason = format_failure_reason(rollout).strip()

    system_prompt = car_dealer_summary_template.render(
        system=True,
        all_car_brands=DEFAULT_BRANDS,
        all_car_types=DEFAULT_TYPES,
    ).strip()

    input_prompt = car_dealer_summary_template.render(
        system=False,
        mode="input",
        chat_history=history_str,
        failure_reason=failure_reason,
    ).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )

    return response, cost

def check_reasoning_feasibility(reason: str) -> Tuple[str, str]:
    """
    Check if the reasoning is infeasible

    Return:
        'high': The reasoning is feasible (didn't have any of the condition that makes it 'medium' or 'low' level of feasibility)
        'medium': The reasoning is potentially feasible, but requires manual verification
        'low': The reasoning is infeasible
    """
    reason_lower = reason.lower()

    if "summary" in reason_lower:
        return 'low'
    else:
        return 'high'

def hindsight_gen_alt_actions(summary: str, 
                              rollout: List[Dict],
                              t: int,
                              num_alt_actions_to_gen: int,
                              buyer_info: dict,
                              car_inventories: dict,
                              max_car: int = 8):
    """
    Generate the alternative actions for the given timestep
    """
    # Build the history
    history = []
    all_api_calls = [] # List[Dict]
    all_api_calls_have_responses = [] # List[bool]
    for j in range(t):
        history.append({
            "role": "seller",
            "content": rollout[j]["action"]
        })
        history.append({
            "role": "buyer",
            "content": rollout[j]["buyer_response"]
        })

        all_api_calls.append(rollout[j]["api_call"])
        all_api_calls_have_responses.append(rollout[j]["api_response"] != [])

    # Step 1: Generate alternative API calls
    history_str = format_chat_history(history)
    past_N = 3 # Hard-coded
    prev_api_call_history = format_api_call_history(all_api_calls, all_api_calls_have_responses, past_N)

    prev_api_call = rollout[t-1]["api_call_used"] if t > 0 else {}

    if t > 0:
        if "car_list" in rollout[t-1]:
            prev_api_response = rollout[t-1]["car_list"]
            prev_api_response_str, _ = format_car_options(prev_api_response, max_car=-1)
        else:
            prev_api_response = rollout[t-1]["api_response_used"]
            prev_api_response_str, _ = format_car_options(prev_api_response, max_car=max_car)
    else:
        prev_api_response = []
        prev_api_response_str, _ = format_car_options(prev_api_response, max_car=max_car)

    system_prompt = car_dealer_api_expert_gen_prefered_api_action_template.render(
        system=True,
        all_car_brands=DEFAULT_BRANDS,
        all_car_types=DEFAULT_TYPES,
        all_car_features=DEFAULT_FEATURES,
        summary=summary,
        past_N=past_N,
    ).strip()

    input_prompt = car_dealer_api_expert_gen_prefered_api_action_template.render(
        system=False,
        mode="input",
        step_idx=t,
        observation_action_history=history_str,
        past_N=past_N,
        prev_api_call_history=prev_api_call_history,
        previous_api_call=prev_api_call,
        previous_api_response=prev_api_response_str,
        num_responses=num_alt_actions_to_gen
    ).strip()

    # print(system_prompt)
    # input("================== API CALL system prompt ==================")
    # print(input_prompt)
    # input("================== API CALL input prompt ==================")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    terminate = False
    query_attempts = 0
    querying_cost = 0

    while not terminate and query_attempts < MAX_QUERY_ATTEMPTS:
        response, cost = generate_from_openai_completion(
            messages=messages, model="gpt-4o"
        )

        querying_cost += cost
        try:
            response_json = parse_json(response)

            assert len(response_json) == num_alt_actions_to_gen, f"Invalid number of alternative actions: {len(response_json)}. Must be {num_alt_actions_to_gen}"
            for i in range(num_alt_actions_to_gen):
                assert response_json[i] is not None, f"Failed to parse the response: {response}"
                assert "teacher_reason" in response_json[i] and "api_call" in response_json[i] and "seller_reason" in response_json[i], f"Invalid response: {response_json[i]}. Must contain 'teacher_reason', 'api_call' and 'seller_reason'"
                assert "api_name" in response_json[i]["api_call"] and "api_brand" in response_json[i]["api_call"] and "api_type" in response_json[i]["api_call"] and "api_features" in response_json[i]["api_call"], f"Invalid API call: {response_json[i]['api_call']}. Must contain 'api_name', 'api_brand', 'api_type' and 'api_features'"
            terminate = True
        except:
            query_attempts += 1
            if query_attempts == MAX_QUERY_ATTEMPTS:
                raise Exception(f"Failed to parse the response: {response}")

        query_attempts += 1

    # print(json.dumps(response_json, indent=4))
    # input("================== API CALL response ==================")

    api_calls = [response_json[i]["api_call"] for i in range(num_alt_actions_to_gen)]
    api_response = []
    for i in range(num_alt_actions_to_gen):
        car_inventory_dict = determine_car_inventory(buyer_info, car_inventories)
        api_response.append(use_api(api_calls[i], car_inventory_dict))

    api_call_used = []
    api_response_used = []
    for i in range(num_alt_actions_to_gen):
        if api_response[i] == []:
            api_call_used.append(prev_api_call)
            api_response_used.append(prev_api_response)
        else:
            api_call_used.append(api_calls[i])
            api_response_used.append(api_response[i])

    # Duplicate the API call twice
    alt_api_reason_action_generated = []
    alt_api_response_generated = []
    for i in range(num_alt_actions_to_gen):
        for _ in range(num_alt_actions_to_gen):
            alt_api_reason_action_generated.append(response_json[i])
            alt_api_response_generated.append(api_response_used[i])

    # Check for feasibility
    for i in range(len(alt_api_reason_action_generated)):
        alt_api_reason_action_generated[i]["feasibility"] = check_reasoning_feasibility(alt_api_reason_action_generated[i]["seller_reason"])

    # Step 2: Generate the alternative responses
    alt_responses_generated = []
    for i in range(num_alt_actions_to_gen):
        car_options_str, car_listed = format_car_options(api_response_used[i], max_car=max_car)

        system_prompt = car_dealer_api_expert_gen_prefered_response_action_template.render(
            system=True,
            all_car_brands=DEFAULT_BRANDS,
            all_car_types=DEFAULT_TYPES,
            summary=summary,
        ).strip()

        input_prompt = car_dealer_api_expert_gen_prefered_response_action_template.render(
            system=False,
            mode="input",
            step_idx=t,
            observation_action_history=history_str,
            api_call_used=api_call_used[i],
            api_response_used=car_options_str,
            num_responses=num_alt_actions_to_gen
        ).strip()

        # print(system_prompt)
        # input("================== RESPONSE system prompt ==================")
        # print(input_prompt)
        # input("================== RESPONSE input prompt ==================")

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
            
            querying_cost += cost
            try:
                response_json = parse_json(response)

                assert len(response_json) == num_alt_actions_to_gen, f"Invalid number of alternative actions: {len(response_json)}. Must be {num_alt_actions_to_gen}"
                for i in range(num_alt_actions_to_gen):
                    assert response_json[i] is not None, f"Failed to parse the response: {response}"
                    assert "teacher_reason" in response_json[i] and "response" in response_json[i] and "car_idx" in response_json[i] and "proposed_car" in response_json[i] and "seller_reason" in response_json[i], f"Invalid response: {response_json[i]}. Must contain 'teacher_reason', 'response', 'car_idx', 'proposed_car' and 'seller_reason'"
                    assert "brand" in response_json[i]["proposed_car"] and "type" in response_json[i]["proposed_car"] and "features" in response_json[i]["proposed_car"] and "msrp" in response_json[i]["proposed_car"], f"Invalid proposed car: {response_json[i]['proposed_car']}. Must contain 'brand', 'type', 'features' and 'msrp'"
                terminate = True
            except:
                query_attempts += 1
                if query_attempts == MAX_QUERY_ATTEMPTS:
                    raise Exception(f"Failed to parse the response: {response}")

            query_attempts += 1

        # print(json.dumps(response_json, indent=4))
        # input("================== RESPONSE response ==================")

        # Set the car chosen by the idx
        for i in range(num_alt_actions_to_gen):
            car_idx = response_json[i]["car_idx"]
            response_json[i]["car_list"] = car_listed
            response_json[i]["car_chosen_by_idx"] = car_listed[car_idx - 1] if car_idx != 0 and car_idx <= len(car_listed) else {}

        alt_responses_generated.extend(response_json)

    # Check for feasibility
    for i in range(len(alt_responses_generated)):
        alt_responses_generated[i]["feasibility"] = check_reasoning_feasibility(alt_responses_generated[i]["seller_reason"])

    # Duplicate the api_call_used and api_response_used
    alt_api_call_used = []
    alt_api_response_used = []
    for i in range(num_alt_actions_to_gen):
        for _ in range(num_alt_actions_to_gen):
            alt_api_call_used.append(api_call_used[i])
            alt_api_response_used.append(api_response_used[i])

    return alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated, querying_cost
    
def generate_rollouts_with_alt_actions(
        rollout_dir: str, 
        n_rollouts_to_sample: int, 
        m_timesteps_to_gen_from: int, 
        actions_to_gen_at_each_timestep: int, 
        rollout_idx_min: int, 
        rollout_idx_max: int,
        data_types: List[str]=["train", "val"],
        start_obj_idx: int=-1, 
        end_obj_idx: int=-1):
    total_cost = 0

    buyer_info_dict = load_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json")
    car_inventories = load_car_inventories()
    
    # Collect the games to generate rollouts for
    for data_type in data_types:
        if data_type == "train":
            buyer_strategy_dict = TRAIN_BUYER_STRATEGIES
            brand_list = TRAIN_BRANDS
            type_list = TRAIN_TYPES
        elif data_type == "val":
            buyer_strategy_dict = VAL_BUYER_STRATEGIES
            brand_list = VAL_BRANDS
            type_list = VAL_TYPES
        elif data_type == "test":
            buyer_strategy_dict = TEST_BUYER_STRATEGIES
            brand_list = TEST_BRANDS
            type_list = TEST_TYPES

        data_type_all_games_to_play_list = []
        for buyer_strategy_id in buyer_strategy_dict.keys():
            for brand in brand_list:
                for car_type in type_list:
                    budget_list = buyer_info_dict[str(buyer_strategy_id)][brand][car_type].keys()
                    for budget in budget_list:
                        buyer_strategy = buyer_strategy_dict[buyer_strategy_id]
                        buyer_info = buyer_info_dict[str(buyer_strategy_id)][brand][car_type][budget]
                        buyer_info["id"] = int(buyer_strategy_id)
                        buyer_info["name"] = buyer_strategy["name"]
                        # Shuffle the feature to avoid overfitting on the first feature
                        random.shuffle(buyer_info["features"])
                        game_id = f"{buyer_strategy_id}_{brand}_{car_type}_{budget}"
                        data_type_all_games_to_play_list.append((game_id, data_type, buyer_info))


        print(f"Evaluating on {len(data_type_all_games_to_play_list)} objects [{start_obj_idx},{end_obj_idx}): {data_type_all_games_to_play_list}")

        summary_dict_path = os.path.join(rollout_dir, data_type, f"_summary_dict_gen_range={rollout_idx_min}-{rollout_idx_max}_si={start_obj_idx}_ei={end_obj_idx}.json")
        if not os.path.exists(summary_dict_path):
            summary_dict = load_json(os.path.join(rollout_dir, data_type, "_summary_dict.json"))  # Assume that the summary dict is already generated
            save_json(summary_dict_path, summary_dict)

        # Get all the rollout files
        json_files = [f for f in os.listdir(os.path.join(rollout_dir, data_type)) if is_valid_rollout(f)]

        # For each task, we sample N rollouts
        for game_id, data_type, buyer_info in tqdm(data_type_all_games_to_play_list, desc="Processing objects"):
            # Get the task specific rollout files
            task_rollout_files = [f for f in json_files if f"{game_id}_" in f and is_within_valid_range(f, rollout_idx_min, rollout_idx_max)]

            # Randomly select N rollouts (no replacement)
            selected_rollouts = random.sample(task_rollout_files, n_rollouts_to_sample)

            print(f"Selected rollouts: {selected_rollouts}")

            rollout_idx = rollout_idx_max

            # For each rollout, we sample M actions
            for rollout_file in selected_rollouts:
                rollout = load_json(os.path.join(rollout_dir, data_type, rollout_file))

                if "summary" not in rollout[0]:
                    summary, cost = gen_summary_from_rollout(rollout)
                    rollout[0]["summary"] = summary
                    save_json(os.path.join(rollout_dir, data_type, rollout_file), rollout)
                    total_cost += cost
                else:
                    summary = rollout[0]["summary"]
                    total_cost += 0

                print(f"Summary for {rollout_file} (cost: {total_cost:.2f}):\n{summary}")

                # Randomly sample M distinct timesteps (After responding to buyer's initial inquiry)
                start_timestep = min(2, len(rollout)-1)
                end_timestep = len(rollout)-1

                if len(range(start_timestep, end_timestep)) < m_timesteps_to_gen_from:
                    start_timestep = 1
                    end_timestep = len(rollout)

                    # If it's still not enough, we will also include the initial timestep
                    if len(range(start_timestep, end_timestep)) < m_timesteps_to_gen_from:
                        start_timestep = 0

                print(f"Rollout file: {rollout_file}, range: {range(start_timestep, end_timestep)}")
                
                timestep_to_gen_from = random.sample(range(start_timestep, end_timestep), m_timesteps_to_gen_from)

                print(f"Timesteps to generate from: {timestep_to_gen_from}")

                for t in timestep_to_gen_from:
                    # Check if the rollout has already been generated
                    summary_dict = load_json(summary_dict_path)
                    if str(rollout_idx) in summary_dict and game_id in summary_dict[str(rollout_idx)]:
                        print(f"Rollout {rollout_idx} for {game_id} has already been generated. Skipping...")
                        rollout_idx += 4 # Skipping 4 because we save 4 files per timestep
                        continue
                    
                    alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated, cost = hindsight_gen_alt_actions(summary, rollout, t, actions_to_gen_at_each_timestep, buyer_info, car_inventories, max_car=8)
                    total_cost += cost
                    alternatives = []
                    for alt_api_reason_action, alt_api_response, alt_api_call_used, alt_api_response_used, alt_response in zip(alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated):
                        alternatives.append(
                            {
                                "api_call": alt_api_reason_action,
                                "api_response": alt_api_response,
                                "api_call_used": alt_api_call_used,
                                "api_response_used": alt_api_response_used,
                                "response": alt_response,
                            }
                        )

                    # Edit the rollout file to save the alt actions
                    if "expert_alternatives" not in rollout[t]:
                        rollout[t]["expert_alternatives"] = []

                    rollout[t]["expert_alternatives"].extend(alternatives)

                    save_json(os.path.join(rollout_dir, data_type, rollout_file), rollout)

                    # Make copy of the rollout file
                    for i in range(actions_to_gen_at_each_timestep * actions_to_gen_at_each_timestep):
                        new_partial_rollout = copy.deepcopy(rollout)
                        new_partial_rollout[0].pop("summary", None)  # Safely remove the summary

                        # Edit the action at t
                        new_partial_rollout[t] = {
                            "step": t,
                            # API call
                            "teacher_api_reason": alternatives[i]["api_call"]["teacher_reason"],
                            "api_reason": alternatives[i]["api_call"]["seller_reason"],
                            "api_call": alternatives[i]["api_call"]["api_call"],
                            "api_response": alternatives[i]["api_response"],
                            "api_feasibility": alternatives[i]["api_call"]["feasibility"],
                            "api_raw_text": "",
                            # API call used
                            "api_call_used": alternatives[i]["api_call_used"],
                            "api_response_used": alternatives[i]["api_response_used"],
                            "car_list": alternatives[i]["response"]["car_list"],
                            # Response
                            "teacher_response_reason": alternatives[i]["response"]["teacher_reason"],
                            "reason": alternatives[i]["response"]["seller_reason"],
                            "action": alternatives[i]["response"]["response"],
                            "car_idx": alternatives[i]["response"]["car_idx"],
                            "proposed_car": alternatives[i]["response"]["car_chosen_by_idx"],
                            "copied_proposed_car": alternatives[i]["response"]["proposed_car"],
                            "response_feasibility": alternatives[i]["response"]["feasibility"],
                            "raw_text": "",  # Because we are using gpt-4o, it has less parsing issues. 
                        }
                        new_partial_rollout = new_partial_rollout[:t+1]

                        # Save the new partial rollout
                        print(f"Saving new partial rollout: {os.path.join(rollout_dir, data_type, f'{game_id}_{rollout_idx}.json')}")

                        save_json(os.path.join(rollout_dir, data_type, f"{game_id}_{rollout_idx}.json"), new_partial_rollout)

                        if alternatives[i]["api_call"]["feasibility"] == "low" or alternatives[i]["response"]["feasibility"] == "low":
                            input("Check for low feasibility. Press Enter to continue...")

                        # Update the summary dict
                        summary_dict = load_json(summary_dict_path)
                        if str(rollout_idx) not in summary_dict:
                            summary_dict[str(rollout_idx)] = []
                        if game_id not in summary_dict[str(rollout_idx)]:
                            summary_dict[str(rollout_idx)].append(game_id)
                        save_json(summary_dict_path, summary_dict)

                        rollout_idx += 1

                    print(f"Total cost: {total_cost:.2f}")
                    # input("Press Enter to continue...")



def prepare_batch(batch_json_files: List[Tuple[str, str, str]], batched_env: BatchedCarDealerEnvironment, buyer_info_dict: Dict, car_inventories: Dict):
    """
    Return:
        - histories: List[List[Dict]]
        - batch_buyer_infos: List[Dict]
        - traj_list: List[List[Dict]]
        - prev_dones: List[bool]
    """
    histories = []
    actions = []
    traj_list = []
    batch_buyer_infos = []

    # From the history
    all_prev_api_calls = [[] for _ in range(len(batch_json_files))]
    all_prev_api_calls_have_responses = [[] for _ in range(len(batch_json_files))]
    num_negotiations = [0 for _ in range(len(batch_json_files))]
    num_car_proposed = [0 for _ in range(len(batch_json_files))]
    
    # Just from the previous timestep
    prev_api_calls = []
    prev_api_responses = []
    prev_proposed_cars = []
    
    proposed_cars = []
    proposed_cars_copied_in_responses = []
    for file, data_type, game_id, rollout_idx in batch_json_files:
        rollout = load_json(os.path.join(rollout_dir, data_type, file))
        traj_list.append(rollout)
        actions.append(rollout[-1]["action"])  # The last action is the expert's action

        rollout_histories = []
        for i in range(len(rollout)-1): # -1 because the last action doesn't have an answer yet
            rollout_histories.append({
                "role": "seller",
                "content": rollout[i]["action"]
            })

            rollout_histories.append({
                "role": "buyer",
                "content": rollout[i]["action"]
            })

            all_prev_api_calls[i].append(rollout[i]["api_call"])
            all_prev_api_calls_have_responses[i].append(rollout[i]["api_response"] != [])

            # Count the number of negotiations and the number of cars proposed
            curr_seller_response = rollout[i]["action"]
            curr_proposed_car = rollout[i]["proposed_car"]
            prev_proposed_car = rollout[i-1]["proposed_car"] if i > 0 else []

            has_discount, _ = check_has_discount(curr_seller_response, curr_proposed_car)
            same_car = are_same_cars(curr_proposed_car, prev_proposed_car)

            if not same_car:
                num_car_proposed[i] += 1
                num_negotiations[i] += 1
            else:
                if has_discount:
                    num_negotiations[i] += 1

        histories.append(rollout_histories)

        buyer_strategy_id, brand, car_type, budget = game_id.split("_")
        buyer_info = buyer_info_dict[buyer_strategy_id][brand][car_type][budget]
        # Shuffle the feature to avoid overfitting on the first feature
        random.shuffle(buyer_info["features"])
        buyer_info["id"] = int(buyer_strategy_id)
        batch_buyer_infos.append(buyer_info)

        # Get the proposed cars
        proposed_cars.append(rollout[-1]["proposed_car"])

        # Get the proposed cars copied in the response
        if "copied_proposed_car" in rollout[-1]:
            proposed_cars_copied_in_responses.append(rollout[-1]["copied_proposed_car"])
        else:
            proposed_cars_copied_in_responses.append([])

        prev_proposed_cars.append(rollout[-2]["proposed_car"] if len(rollout) > 1 else [])

        # prev_api_call
        prev_api_calls.append(rollout[-1]["api_call_used"])
        # prev_api_response
        if "car_list" in rollout[-1]:
            prev_api_responses.append(rollout[-1]["car_list"])
        else:
            prev_api_responses.append(rollout[-1]["api_response_used"])


    # for i in range(len(histories)):
    #     print(prev_api_responses[i])
    #     print(f"file: {batch_json_files[i][0]}")
    #     input(f"=========== {i} ===========")

    prev_dones = [False for _ in range(len(histories))]
    # Take a step in the environment using expert's action
    histories, buyer_reasons, buyer_responses, buyer_decisions, rewards, successes, failure_reasons, dones, num_negotiations, prev_proposed_cars, num_car_proposed = batched_env.step(batch_buyer_infos, histories, actions, proposed_cars, proposed_cars_copied_in_responses, num_negotiations, prev_proposed_cars, num_car_proposed, car_inventories, prev_dones)

    # Log the trajectories
    for i in range(len(batch_json_files)):
        if not prev_dones[i]:
            traj_list[i][-1]["buyer_reason"] = buyer_reasons[i]
            traj_list[i][-1]["buyer_response"] = buyer_responses[i]
            traj_list[i][-1]["buyer_decision"] = buyer_decisions[i]
            traj_list[i][-1]["reward"] = rewards[i]
            traj_list[i][-1]["success"] = successes[i]
            traj_list[i][-1]["failure_reason"] = failure_reasons[i]
            traj_list[i][-1]["api_score"] = None 
            traj_list[i][-1]["score"] = None  # Critic is not used to score the expert's action

    prev_dones = dones

    return histories, batch_buyer_infos, traj_list, prev_dones, all_prev_api_calls, all_prev_api_calls_have_responses,  num_negotiations, num_car_proposed, prev_api_calls, prev_api_responses, prev_proposed_cars


def complete_rollouts(sim_host: str, sim_port: int, rollout_dir: str, agent_config: Dict, bs: int, rollout_idx_min: int=-1, rollout_idx_max: int=-1, data_types: List[str]=["train", "val"], start_obj_idx: int=-1, end_obj_idx: int=-1):
    """
    Complete the rollouts
    """
    # Initialize the agent
    print(f"Initializing agent {agent_config['log_name']}, {agent_config['model_id']}")
    agent = initialize_agent(agent_config,
                            parse_reason_action_fn=parse_reason_and_action_car_dealer,
                            verbose=agent_config["verbose"],
                            debug=agent_config["debug"])
    batched_env = setup_batched_car_dealer_env(host=sim_host,
                                                     port=sim_port)
    
    with open("prompts/car_dealer/car_dealer_api_template.j2", "r") as f:
        agent_api_call_template = Template(f.read())

    with open("prompts/car_dealer/car_dealer_template.j2", "r") as f:
        agent_prompt_template = Template(f.read())

    car_inventories = load_car_inventories()
    buyer_info_dict = load_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json")

    # Collect all the rollouts that need to be completed
    json_files_to_complete = []

    for data_type in data_types:
        summary_dict = load_json(os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json"))

        json_files = [(f, data_type, get_game_id_and_rollout_idx_from_file_name(f)[0], get_game_id_and_rollout_idx_from_file_name(f)[1]) for f in os.listdir(os.path.join(rollout_dir, data_type)) if is_valid_rollout(f) and need_to_complete(f, summary_dict) and is_within_valid_range(f, rollout_idx_min, rollout_idx_max)]

        json_files_to_complete.extend(json_files)

        # Save a copy of the summary dict
        summary_dict_path = os.path.join(rollout_dir, data_type, f"_rollout_complete_summary_dict_range={rollout_idx_min}-{rollout_idx_max}_si={start_obj_idx}_ei={end_obj_idx}.json")
        if not os.path.exists(summary_dict_path):
            save_json(summary_dict_path, summary_dict)
        else:
            summary_dict = load_json(summary_dict_path)

    # Filter out the rollouts that are already completed
    filtered_json_files_to_complete = []
    for file, data_type, game_id, rollout_idx in json_files_to_complete:
        if str(rollout_idx) not in summary_dict or game_id not in summary_dict[str(rollout_idx)]:

            filtered_json_files_to_complete.append((file, data_type, game_id, rollout_idx))
    json_files_to_complete = filtered_json_files_to_complete

    print(f"Total number of rollouts to complete: {json_files_to_complete}\nlen: {len(json_files_to_complete)}")
    # input("stop")

    for batch in tqdm(range(math.ceil(len(json_files_to_complete) / bs))):
        batch_json_files = json_files_to_complete[batch * bs:(batch + 1) * bs]

        print(f"Batch {batch} has {len(batch_json_files)} rollouts: {batch_json_files}")

        # Prepare the batch histories, batch_buyer_infos, traj_list, prev_dones, all_prev_api_calls, all_prev_api_calls_have_responses,  num_negotiations, num_car_proposed, prev_api_calls, prev_api_responses, prev_proposed_cars
        histories, batch_buyer_infos, traj_list, prev_dones, all_prev_api_calls, all_prev_api_calls_have_responses,  num_negotiations, num_car_proposed, prev_api_calls, prev_api_responses, prev_proposed_cars = prepare_batch(batch_json_files, batched_env, buyer_info_dict, car_inventories)

        # Rollout the batch
        traj_list = rollout_batch(agent_api_call_template, agent_prompt_template, 
                                  agent, batched_env, car_inventories, batch_buyer_infos, 
                                  histories, traj_list, prev_dones, NUM_ALT_RESPONSES, 
                                  # From the history
                                  all_prev_api_calls_in=all_prev_api_calls,
                                  all_prev_api_calls_have_responses_in=all_prev_api_calls_have_responses,
                                  num_negotiations_in=num_negotiations,
                                  num_car_proposed_in=num_car_proposed,
                                  # From the last timestep
                                  prev_api_calls_in=prev_api_calls,
                                  prev_api_responses_in=prev_api_responses,
                                  prev_proposed_cars_in=prev_proposed_cars,
                                  )

        # for i in range(len(batch_json_files)):
        #     for t in range(len(traj_list[i])):
        #         print(json.dumps(traj_list[i][t], indent=4))
        #         if t != 0:
        #             if "car_list" in traj_list[i][t]:
        #                 print(traj_list[i][t]["car_list"])
        #             else:
        #                 print(traj_list[i][t]["api_response_used"])
        #             print(traj_list[i][t]["raw_text"])
        #             print(traj_list[i][t]["proposed_car"])
        #         print(batch_json_files[i][0])
        #         input(f"traj_list {i} t={t}")
        #     input(f"=========== traj_list {i} ===========")

        # Save the trajectories
        for i in range(len(batch_json_files)):
            data_type = batch_json_files[i][1]
            file = batch_json_files[i][0]
            save_json(os.path.join(rollout_dir, data_type, file), traj_list[i])

            game_id, rollout_idx = get_game_id_and_rollout_idx_from_file_name(file)

            summary_dict = load_json(summary_dict_path)
            if rollout_idx not in summary_dict:
                summary_dict[rollout_idx] = []

            summary_dict[rollout_idx].append(game_id)
            save_json(summary_dict_path, summary_dict)

        

def merge_rollout_summary_dicts(rollout_dir: str, data_types: List[str]) -> Dict:
    """
    Merge the summary dicts
    """
    for data_type in data_types:
        original_rollout_summary_dict_path = os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json")
        rollout_summary_dicts_path = [f for f in os.listdir(os.path.join(rollout_dir, data_type)) if "_rollout_complete_summary_dict" in f and  "_rollout_complete_summary_dict.json" not in f and "copy" not in f]
        rollout_summary_dicts_path.sort()
        print(f"Rollout summary dicts path: {rollout_summary_dicts_path}")
        input("stop")
        
        main_summary_dict = load_json(original_rollout_summary_dict_path)

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            summary_dict = load_json(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))

            for rollout_idx, obj_list in summary_dict.items():
                if rollout_idx not in main_summary_dict:
                    main_summary_dict[rollout_idx] = []

                for obj in obj_list:
                    if obj not in main_summary_dict[rollout_idx]:
                        main_summary_dict[rollout_idx].append(obj)

        # Sort the summary dict by the rollout idx
        main_summary_dict = dict(sorted(main_summary_dict.items(), key=lambda x: int(x[0])))

        print(f"Main summary dict keys: {main_summary_dict.keys()}")
        print(f"Main summary value len: {[len(v) for v in main_summary_dict.values()]}")
        input("Check the merged summary dict before we save it")
        
        save_json(os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json"), main_summary_dict)

        input("Please check the merged summary dict before we delete the intermediate summary dicts")

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            print(f"Deleting {os.path.join(rollout_dir, data_type, rollout_summary_dict_path)}")
            input("stop")
            os.remove(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))


def merge_generation_summary_dicts(rollout_dir: str, data_types: List[str]) -> Dict:
    """
    Merge the summary dicts
    """
    for data_type in data_types:
        original_rollout_summary_dict_path = os.path.join(rollout_dir, data_type, "_summary_dict.json")
        rollout_summary_dicts_path = [f for f in os.listdir(os.path.join(rollout_dir, data_type)) if "_summary_dict" in f and  "rollout_complete" not in f and "copy" not in f and "_summary_dict.json" not in f]
        rollout_summary_dicts_path.sort()
        print(f"Rollout summary dicts path: {rollout_summary_dicts_path}")
        input("stop")
        
        main_summary_dict = load_json(original_rollout_summary_dict_path)
        # Save a copy for rollout_complete
        save_json(os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json"), main_summary_dict)
        input("Saved the rollout complete summary dict")

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            summary_dict = load_json(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))

            for rollout_idx, obj_list in summary_dict.items():
                if rollout_idx not in main_summary_dict:
                    main_summary_dict[rollout_idx] = []

                for obj in obj_list:
                    if obj not in main_summary_dict[rollout_idx]:
                        main_summary_dict[rollout_idx].append(obj)

        # Sort the summary dict by the rollout idx
        main_summary_dict = dict(sorted(main_summary_dict.items(), key=lambda x: int(x[0])))

        print(f"Main summary dict keys: {main_summary_dict.keys()}")
        print(f"Main summary value len: {[len(v) for v in main_summary_dict.values()]}")
        input("Check the merged summary dict before we save it")
        
        save_json(os.path.join(rollout_dir, data_type, "_summary_dict.json"), main_summary_dict)

        input("Please check the merged summary dict before we delete the intermediate summary dicts")

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            print(f"Deleting {os.path.join(rollout_dir, data_type, rollout_summary_dict_path)}")
            input("stop")
            os.remove(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, choices=list(range(3)), required=True)
    parser.add_argument("-m", "--mode", type=str, choices=["g", "gen_actions", "r", "rollout", "merge_gen", "merge_rollout"], required=True)
    parser.add_argument("-d", "--data_types", help="List of data types to process", nargs="+", choices=["train", "val"])
    parser.add_argument("-e", "--elogger", action="store_true", default=False)
    parser.add_argument("--n_rollouts_to_sample", type=int, default=2)
    parser.add_argument("--m_timesteps_to_gen_from", type=int, default=2)
    parser.add_argument("--actions_to_gen_at_each_timestep", type=int, default=2)
    parser.add_argument("--starting_num_rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-min", "--rollout_idx_min", type=int, default=-1, help="When it's generating rollouts, define the range of rollout idx to sample from. When it's completing rollouts, define the range of rollout idx to complete")
    parser.add_argument("-max", "--rollout_idx_max", type=int, default=-1)
    parser.add_argument("-ns", "--no_serve_model", action="store_true", default=False)
    parser.add_argument("-p", "--port", type=str, default=None)
    parser.add_argument("-si", "--start_obj_idx", type=int, default=-1, help="Within a data_type, start from this object idx")
    parser.add_argument("-ei", "--end_obj_idx", type=int, default=-1, help="Within a data_type, end at this object idx")
    parser.add_argument("--sim_host", type=str, default="localhost")
    parser.add_argument("--sim_port", type=int, default=40042)
    args = parser.parse_args()
    random.seed(args.seed)

    elogger.set_activate(args.elogger)

    rollout_dir = iter_to_rollout_dir[args.iter]
    if args.mode == "g" or args.mode == "gen_actions":
        print(f"Generating rollouts with alt actions for data_type={args.data_types} in {rollout_dir}")

        try:
            generate_rollouts_with_alt_actions(
                rollout_dir=rollout_dir, 
                n_rollouts_to_sample=args.n_rollouts_to_sample, 
                m_timesteps_to_gen_from=args.m_timesteps_to_gen_from, 
                actions_to_gen_at_each_timestep=args.actions_to_gen_at_each_timestep, 
                rollout_idx_min=args.rollout_idx_min, 
                rollout_idx_max=args.rollout_idx_max,
                data_types=args.data_types, 
                start_obj_idx=args.start_obj_idx, 
                end_obj_idx=args.end_obj_idx)
        except Exception as e:
            elogger.log(f"Error generating rollouts with alt actions: {e}")
            raise e

        elogger.log(f"Successfully generated rollouts with alt actions for data_type={args.data_types} in {rollout_dir}")
    elif args.mode == "r" or args.mode == "rollout":
        agent_config = iter_to_agent_config[args.iter]

        if not args.no_serve_model:
            # Example of dist_url_port: 29540
            process, server_url, base_gpu_id = start_sglang_server(model_path=agent_config["model_id"],
                                                    port=args.port, 
                                                    tp=1,
                                                    dist_url_port=agent_config["dist_url_port"])
            agent_config["server_url"] = server_url

        complete_rollouts(sim_host=args.sim_host, sim_port=args.sim_port, rollout_dir=rollout_dir, agent_config=agent_config, bs=agent_config["batch_limit"], rollout_idx_min=args.rollout_idx_min, rollout_idx_max=args.rollout_idx_max, data_types=args.data_types, start_obj_idx=args.start_obj_idx, end_obj_idx=args.end_obj_idx)

        elogger.log(f"Successfully completed rollouts for data_type={args.data_types} in {rollout_dir}")
    elif args.mode == "merge_gen":
        merge_generation_summary_dicts(rollout_dir=rollout_dir, data_types=args.data_types)
    elif args.mode == "merge_rollout":
        merge_rollout_summary_dicts(rollout_dir=rollout_dir, data_types=args.data_types)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")