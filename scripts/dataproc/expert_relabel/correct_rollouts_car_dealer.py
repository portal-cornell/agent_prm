"""
Correct all the states and actions of a failed rollout

for each failed rollout,
    - generate a summary
    - generate a correction for each state and action (given state and the summary)

Usage:
    python scripts/dataproc/expert_relabel/correct_rollouts_car_dealer.py -d train -i 1 -e

    -i iteration, the iteration for the pi that you are training for (e.g., pi3 would be iteration=2)
"""
import argparse
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict, Tuple
from jinja2 import Template
from omegaconf import DictConfig, OmegaConf

from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json

from agent_prm.envs.car_dealer.parser import parse_reason_and_action_car_dealer
from agent_prm.envs.car_dealer.data import TRAIN_BUYER_STRATEGIES, VAL_BUYER_STRATEGIES, TEST_BUYER_STRATEGIES, TRAIN_BRANDS, VAL_BRANDS, TEST_BRANDS, TRAIN_TYPES, VAL_TYPES, TEST_TYPES, DEFAULT_BRANDS, DEFAULT_TYPES, DEFAULT_FEATURES, format_chat_history, format_car_options, format_api_call_history, determine_car_inventory, load_car_inventories, check_has_discount
from agent_prm.envs.car_dealer.interface import use_api

MAX_QUERY_ATTEMPTS = 3

with open("prompts/car_dealer/car_dealer_summary_history_template.j2", "r") as f:
    car_dealer_summary_history_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_summary.j2", "r") as f:
    car_dealer_summary_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_api_expert_gen_prefered_api_action.j2", "r") as f:
    car_dealer_api_expert_gen_prefered_api_action_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_api_expert_gen_prefered_response_action.j2", "r") as f:
    car_dealer_api_expert_gen_prefered_response_action_template = Template(f.read())

def relabel_one_rollout(rollout_path: str, buyer_info: dict, car_inventories: dict):
    """
    Effect:
        - Generate a summary of the rollout
        - Generate a correction for each state and action (given state and the summary)
    """
    rollout = load_json(rollout_path)
    total_cost = 0

    # Generate a summary
    if "summary" not in rollout[0]:
        summary, cost = gen_summary_from_rollout(rollout)
        rollout[0]["summary"] = summary
        save_json(rollout_path, rollout)
        total_cost += cost
    else:
        summary = rollout[0]["summary"]
        total_cost += 0

    print(f"=========== Summary ===========")
    print(summary)
    print("="*100)

    # Iterate through the rollout and generate a correction for each state and action
    for t in range(len(rollout)):
        if "expert_alternatives" not in rollout[t]:
            rollout[t]["expert_alternatives"] = []  # Edit the rollout file to save the alt actions

            alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated, cost = hindsight_gen_alt_actions(summary, rollout, t, 1, buyer_info, car_inventories, max_car=8)

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

            print(f"t={t}: Alt reason actions (cost: ${cost:.2f}):\n{json.dumps(alternatives, indent=4)}")

            rollout[t]["expert_alternatives"].extend(alternatives)

            save_json(rollout_path, rollout)
            total_cost += cost
        else:
            print(f"t={t}: Skip because already has expert alternatives")

        print(rollout_path) # So it's easier to debug and find the rollout

    print(f"Total cost = ${total_cost} for {rollout_path}")

    return total_cost

def format_chat_history_and_goal(rollout: List[Dict], t: int, secret_word:str, category: str="") -> Tuple[str, str]:
    """
    Return
        - chat_history: str (until t)
        - goal: str (optional, if file_name is provided)
    """
    answer_str = f"The secret word is '{secret_word}'. It's in the general category '{category}'."

    history_str = ""
    for i in range(t):
        history_str += f"Question #{i+1}: {rollout[i]['action']}\nAnswer #{i+1}: {rollout[i]['answer']}\n"

    if history_str == "":
        history_str = "No chat history yet. Just start with the question."

    return history_str, answer_str

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

        if alt_api_reason_action_generated[i]["feasibility"] == "low":
            print(f"=======================")
            print(history_str)
            print(f"Teacher Reasoning:\n{alt_api_reason_action_generated[i]['teacher_reason']}")
            print(f"API Call:\n{alt_api_reason_action_generated[i]['api_call']}")
            print(f"Player Reasoning:\n{alt_api_reason_action_generated[i]['seller_reason']}")
            print(f"Feasibility: {alt_api_reason_action_generated[i]['feasibility']}")
            print(f"=======================")
            elogger.log(f"[Gen API Call] Please verify the reasoning of the following rollout: {alt_api_reason_action_generated[i]}")
            input("Press Enter to continue...")

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

        if alt_responses_generated[i]["feasibility"] == "low":
            print(f"=======================")
            print(history_str)
            print(f"Teacher Reasoning:\n{alt_responses_generated[i]['teacher_reason']}")
            print(f"Response:\n{alt_responses_generated[i]['response']}")
            print(f"Player Reasoning:\n{alt_responses_generated[i]['seller_reason']}")
            print(f"Feasibility: {alt_responses_generated[i]['feasibility']}")
            print(f"=======================")
            elogger.log(f"[Gen Response] Please verify the reasoning of the following rollout: {alt_responses_generated[i]}")
            input("Press Enter to continue...")

    # Duplicate the api_call_used and api_response_used
    alt_api_call_used = []
    alt_api_response_used = []
    for i in range(num_alt_actions_to_gen):
        for _ in range(num_alt_actions_to_gen):
            alt_api_call_used.append(api_call_used[i])
            alt_api_response_used.append(api_response_used[i])

    return alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated, querying_cost

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

def get_failed_rollouts(config: DictConfig, data_types: List[str], iteration: int, buyer_info_dict: dict):
    """
    Returns:
        - a list of failed rollout path
    """
    rollout_dir = config.leap[f"iter{iteration - 1}"]["rollout_dir"]
    
    failed_rollout_paths = []

    # Get the failed rollouts (and the object and the category)
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
                        for rollout_idx in range(config.leap.rollout_per_task_range_min, config.leap.rollout_per_task_range_max):
                            buyer_strategy = buyer_strategy_dict[buyer_strategy_id]
                            buyer_info = buyer_info_dict[str(buyer_strategy_id)][brand][car_type][budget]
                            buyer_info["id"] = int(buyer_strategy_id)
                            buyer_info["name"] = buyer_strategy["name"]
                            # Shuffle the feature to avoid overfitting on the first feature
                            random.shuffle(buyer_info["features"])
                            game_id = f"{buyer_strategy_id}_{brand}_{car_type}_{budget}"
                            data_type_all_games_to_play_list.append((game_id, data_type, buyer_info, rollout_idx))

        for game_id, data_type, buyer_info, rollout_idx in tqdm(data_type_all_games_to_play_list, desc="Processing games"):
            rollout_path = os.path.join(rollout_dir, data_type, f"{game_id}_{rollout_idx}.json")
            if is_failed_rollout(rollout_path):
                failed_rollout_paths.append((rollout_path, buyer_info))

    return failed_rollout_paths


def is_failed_rollout(rollout_path: str):
    """
    Effect:
        - Check if the rollout is failed
    """
    rollout = load_json(rollout_path)
    return not rollout[-1]["success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/create_sft_training_data/car_dealer.yaml")
    parser.add_argument("-d", "--data_types", help="List of data types to process", nargs="+", choices=["train", "val"])
    parser.add_argument("-i", "--iteration", type=int, default=0, help="The iteration number")
    parser.add_argument("-e", "--elogger", action="store_true", default=False, help="Whether to send email alerts")
    args = parser.parse_args()

    elogger.set_activate(args.elogger)

    # Load the config
    config = OmegaConf.load(args.config)

    buyer_info_dict = load_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json")
    car_inventories = load_car_inventories()

    # Get all the failed rollouts
    failed_rollout_paths = get_failed_rollouts(config, args.data_types, args.iteration, buyer_info_dict)

    print(f"failed_rollout_paths: {failed_rollout_paths}")
    print(f"======= Correcting {len(failed_rollout_paths)} failed rollouts")
    input("Press Enter to continue...")

    total_cost = 0
    # Correct the rollouts
    for failed_rollout_path, buyer_info in tqdm(failed_rollout_paths, desc="Correcting rollouts"):
        print(f"\nCorrecting {failed_rollout_path}")
        total_cost += relabel_one_rollout(failed_rollout_path, buyer_info, car_inventories)
        print(f"====== Total cost so far: ${total_cost:.2f} ======")

    elogger.log(f"Successfully corrected {len(failed_rollout_paths)} failed rollouts (total cost: ${total_cost:.2f})")