"""
python scripts/examples/test_hindsight_car_dealer.py
"""

import json
import os
from typing import List, Dict
from jinja2 import Template

from agent_prm.envs.car_dealer.data import DEFAULT_BRANDS, DEFAULT_TYPES, DEFAULT_FEATURES, format_chat_history, format_car_options, format_api_call_history, determine_car_inventory, load_car_inventories
from agent_prm.envs.car_dealer.parser import parse_reason_and_action_car_dealer
from agent_prm.envs.car_dealer.interface import use_api
from agent_prm.utils.general_utils import load_json
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json

rollout_path_list = [
    "REDACTED"
]

MAX_QUERY_ATTEMPTS = 3
with open("prompts/car_dealer/car_dealer_summary_history_template.j2", "r") as f:
    car_dealer_summary_history_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_summary.j2", "r") as f:
    car_dealer_summary_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_api_expert_gen_prefered_api_action.j2", "r") as f:
    car_dealer_api_expert_gen_prefered_api_action_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_api_expert_gen_prefered_response_action.j2", "r") as f:
    car_dealer_api_expert_gen_prefered_response_action_template = Template(f.read())


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
    prev_api_response = rollout[t-1]["api_response_used"] if t > 0 else []

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
        previous_api_response=format_car_options(prev_api_response, max_car=max_car)[0],
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

    # Step 2: Generate the alternative responses
    alt_responses_generated = []
    for i in range(num_alt_actions_to_gen):
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
            api_response_used=format_car_options(api_response_used[i], max_car=max_car)[0],
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

        alt_responses_generated.extend(response_json)

    # Duplicat the api_call_used and api_response_used
    alt_api_call_used = []
    alt_api_response_used = []
    for i in range(num_alt_actions_to_gen):
        for _ in range(num_alt_actions_to_gen):
            alt_api_call_used.append([api_call_used[i]])
            alt_api_response_used.append([api_response_used[i]])

    return alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated


def main():
    car_inventories = load_car_inventories()
    buyer_info_dict = load_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json")

    for rollout_path in rollout_path_list:
        with open(rollout_path, "r") as f:
            rollout = json.load(f)

        game_id = rollout_path.split("/")[-1].split(".")[0]
        summary_path = f"playground/car_dealer/hindsight/{game_id}.txt"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                summary_str = f.read()
        else:
            summary_str = gen_summary_from_rollout(rollout)
            
            # Save the summary to a file
            with open(f"playground/car_dealer/hindsight/{game_id}.txt", "w") as f:
                f.write(summary_str)

            input("Generated summary. Press Enter to continue.")

        # Make a copy of the rollout to playground
        rollout_path = f"playground/car_dealer/hindsight/{game_id}.json"
        with open(rollout_path, "w") as f:
            json.dump(rollout, f, indent=4)
        
        buyer_strategy_id, brand, car_type, budget, _ = game_id.split("_")
        buyer_info = buyer_info_dict[buyer_strategy_id][brand][car_type][budget]
        buyer_info["id"] = int(buyer_strategy_id)

        for i in range(2, len(rollout) - 1):
            # Generate the prompt for the API expert
            alt_api_reason_action_generated, alt_api_response_generated, alt_api_call_used, alt_api_response_used, alt_responses_generated = hindsight_gen_alt_actions(summary_str, rollout, i, 2, buyer_info, car_inventories)

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

            # Log the rollout
            if "expert_alternatives" not in rollout[i]:
                rollout[i]["expert_alternatives"] = []

            rollout[i]["expert_alternatives"].extend(alternatives)

            # Save the rollout
            with open(rollout_path, "w") as f:
                json.dump(rollout, f, indent=4)

        input("================== Press Enter to continue ==================")
    
    
if __name__ == "__main__":
    main()
