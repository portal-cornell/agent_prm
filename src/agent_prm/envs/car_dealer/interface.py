import os
import copy
import time
import json
from jinja2 import Template
from typing import List

from agent_prm.agents.agent import Agent
from agent_prm.envs.car_dealer.data import DEFAULT_BRANDS, DEFAULT_TYPES, DEFAULT_FEATURES
from agent_prm.envs.car_dealer.data import format_chat_history, format_car_options, format_api_call_history, determine_car_inventory, format_most_recent_buyer_message
from agent_prm.envs.car_dealer.parser import parse_reason_and_action_car_dealer_api_call, parse_reason_and_action_car_dealer

def use_api(arguments: dict, car_inventory_dict: dict):
    """
    Return a list of cars that match the given arguments.
    """
    api_brand = arguments["api_brand"].strip().lower().capitalize()
    if api_brand.lower() == "mercedes-benz":
        api_brand = "Mercedes-benz"
    elif api_brand.lower() == "bmw":
        api_brand = "Bmw"
        
    api_type = arguments["api_type"].strip().lower()
    if api_type == "suv":
        api_type = "SUV"  # The database uses "SUV" instead of "suv"

    try:
        api_features = [feature.strip().lower() for feature in arguments["api_features"]]
    except Exception as e:
        api_features = []

    try:
        if arguments["api_name"] == "search_car_by_brand_type":
            return search_car_by_brand_type(api_brand, api_type, car_inventory_dict)
        elif arguments["api_name"] == "search_car_by_brand":
            return search_car_by_brand(api_brand, car_inventory_dict)
        elif arguments["api_name"] == "search_car_by_type":
            return search_car_by_type(api_type, car_inventory_dict)
        elif arguments["api_name"] == "search_car_that_have_features":
            return search_car_that_have_features(api_features, car_inventory_dict)
        elif arguments["api_name"] == "no_op":
            return []
        else:
            # print(f"Invalid API name: {arguments['api_name']}")
            return []
    except Exception as e:
        # print(f"Error in use_api: {e}")
        return []

def search_car_by_brand_type(brand: str, car_type: str, car_inventory_dict: dict):
    if brand not in car_inventory_dict:
        return []

    if car_type not in car_inventory_dict[brand]:
        return []

    car_list = copy.deepcopy(car_inventory_dict[brand][car_type])

    # Add the car's brand and car type to the car list
    for car in car_list:
        car["brand"] = brand
        car["type"] = car_type

    return car_list

def search_car_by_brand(brand: str, car_inventory_dict: dict):
    car_list = []
    for car_type in car_inventory_dict[brand]:
        for car in car_inventory_dict[brand][car_type]:
            car_info = copy.deepcopy(car)
            # Add the car's brand and car type to the car info
            car_info["brand"] = brand
            car_info["type"] = car_type
            car_list.append(car_info)
    return car_list

def search_car_by_type(car_type: str, car_inventory_dict: dict):
    car_list = []
    for brand in car_inventory_dict:
        if car_type in car_inventory_dict[brand]:
            for car in car_inventory_dict[brand][car_type]:
                car_info = copy.deepcopy(car)
                car_info["brand"] = brand
                car_info["type"] = car_type
                car_list.append(car_info)
    return car_list

def search_car_that_have_features(features: List[str], car_inventory_dict: dict):
    car_list = []
    for brand in car_inventory_dict:
        for car_type in car_inventory_dict[brand]:
            for car in car_inventory_dict[brand][car_type]:
                car_features = sorted(car["features"])
                car_features = [feature.strip().lower() for feature in car_features]
                if all([feature in car_features for feature in features]):
                    car_info = copy.deepcopy(car)
                    car_info["brand"] = brand
                    car_info["type"] = car_type
                    car_list.append(car_info)
    return car_list


def query_agent_batch(agent_api_call_template: Template, agent_prompt_template: Template, agent: Agent, histories: List[List[dict]], all_prev_api_calls: List[List[dict]], all_prev_api_calls_have_responses: List[List[bool]], prev_api_calls: List[dict], prev_api_responses: List[dict], num_alt_responses: int, buyer_infos: List[dict], car_inventories: dict, past_N: int = 3):
    """
    Query the agent in batch

    Parameters:
        past_N: int
            The number of previous API calls to include in the prompt.

    Returns:
        api_reasons_actions_dict: List[List[dict]]
            # Game x (1 + num_alt_responses) responses
            Each response is a dict with fields:
                "reason": str
                "action": dict
                "score": float (optional)
        generated_api_texts: List[str]
        api_responses_raw: List[List[dict]]
        api_calls_used_raw: List[dict]
        api_responses_used_raw: List[dict]
        reasons_actions_dict: List[List[dict]]
        proposed_cars_raw: List[dict]
        proposed_cars_copied_in_response_raw: List[dict]
        generated_raw_texts: List[str]  
    """
    if "max-car-8" in agent.name():
        max_car = 8
    else:
        # Show all the cars
        max_car = -1

    formated_histories = [
        format_chat_history(history) for history in histories
    ]
    buyer_responses = [
        format_most_recent_buyer_message(history) for history in histories
    ]
    formatted_prev_api_responses = [
        format_car_options(prev_api_response, max_car=max_car)[0] for prev_api_response in prev_api_responses
    ]
    formatted_prev_api_call_histories= [
        format_api_call_history(all_prev_api_call_list, all_prev_api_calls_have_response_list, past_N) for all_prev_api_call_list, all_prev_api_calls_have_response_list in zip(all_prev_api_calls, all_prev_api_calls_have_responses)
    ]

    ### Step 1: Query the API
    if "dual_agents" in agent.name():
        agent.api_caller.set_prompt_template(prompt_template=agent_api_call_template)
        agent.api_caller.set_parse_reason_action_fn(parse_reason_action_fn=parse_reason_and_action_car_dealer_api_call)
    else:
        agent.set_prompt_template(prompt_template=agent_api_call_template)
        agent.set_parse_reason_action_fn(parse_reason_action_fn=parse_reason_and_action_car_dealer_api_call)

    input_datas = [
        {
            "mode": "input",
            "all_car_brands": DEFAULT_BRANDS,
            "all_car_types": DEFAULT_TYPES,
            "all_car_features": DEFAULT_FEATURES,
            "observation_action_history": formated_history,
            'past_N': past_N,
            'prev_api_call_history': formatted_prev_api_call_history,
            "previous_api_call": prev_api_call,
            "previous_api_response": formatted_prev_api_response
        } for formated_history, formatted_prev_api_call_history, prev_api_call, formatted_prev_api_response in zip(formated_histories, formatted_prev_api_call_histories, prev_api_calls, formatted_prev_api_responses)
    ]

    if "dual_agents" in agent.name():
        reason_actions_api, generated_api_texts = agent.predict_reason_action_batch(
            "api_call",
            input_datas,
            num_responses=1 + num_alt_responses,
            alt_temperature_for_extra_responses=1.0 if num_alt_responses > 0 else None
        ) # List[List[dict]], List[List[str]]
    else:
        reason_actions_api, generated_api_texts = agent.predict_reason_action_batch(
            input_datas,
            num_responses=1 + num_alt_responses,
            alt_temperature_for_extra_responses=1.0 if num_alt_responses > 0 else None
        ) # List[List[dict]], List[List[str]]
    # for i in range(len(reason_actions_api)):
    #     for j in range(1 + num_alt_responses):
    #         print(generated_api_texts[i * (1 + num_alt_responses) + j])
    #         print(reason_actions_api[i][j])
    #         print(f"========= reason_actions_api {i} {j} =========")
            # input(f"========= reason_actions_api {i} {j} =========")
    
    # Parse the API responses and call the API
    api_calls = [[response["action"] for response in game_responses] for game_responses in reason_actions_api] # List[List[dict]]

    api_responses = []
    for i in range(len(api_calls)):
        # Determine the car inventory to use
        car_inventory_dict = determine_car_inventory(buyer_infos[i], car_inventories)
        api_responses.append([use_api(api_call, car_inventory_dict) for api_call in api_calls[i]]) # List[List[List[dict]]] (for each game, for each response, List[dict])

    # Determine the API responses to use
    api_calls_used = [] # List[dict] (one per game, unlike above where there are multiple per game)
    api_responses_used = [] # List[dict] (one per game, unlike above where there are multiple per game)
    # Iterate over the games
    for i in range(len(api_calls)):
        # print(api_calls[i][0])
        # print(api_responses[i][0])
        # print(f"========= actual api call and response {i} =========")
        # input(f"========= actual api call and response {i} =========")
        # Assume that the first API call is the one that is used
        if api_responses[i][0] == []:
            api_calls_used.append(prev_api_calls[i])
            api_responses_used.append(prev_api_responses[i])
        else:
            api_calls_used.append(api_calls[i][0])
            api_responses_used.append(api_responses[i][0])

        # print(api_calls_used[i])
        # print(api_responses_used[i])
        # print(f"========= api_calls_used and api_responses_used {i} =========")
        # input(f"========= api_calls_used and api_responses_used {i} =========")

    ### Step 2: Talk to the user based on the API responses
    if "dual_agents" in agent.name():
        agent.response_generator.set_prompt_template(prompt_template=agent_prompt_template)
        agent.response_generator.set_parse_reason_action_fn(parse_reason_action_fn=parse_reason_and_action_car_dealer)
    else:
        agent.set_prompt_template(prompt_template=agent_prompt_template)
        agent.set_parse_reason_action_fn(parse_reason_action_fn=parse_reason_and_action_car_dealer)
    
    car_lists = []
    api_responses_used_str = []
    for i in range(len(api_responses_used)):
        api_response_str, car_list = format_car_options(api_responses_used[i], max_car=max_car)
        car_lists.append(car_list)
        api_responses_used_str.append(api_response_str)

    input_datas = [
        {
            "mode": "input",
            "all_car_brands": DEFAULT_BRANDS,
            "all_car_types": DEFAULT_TYPES,
            "observation_action_history": formated_history,
            "api_call": api_call_used,
            "api_response": api_response_str,
            "buyer_response": buyer_response
        } for formated_history, api_call_used, api_response_str, buyer_response in zip(formated_histories, api_calls_used, api_responses_used_str, buyer_responses)
    ]

    if "dual_agents" in agent.name():
        reason_actions, generated_texts = agent.predict_reason_action_batch(
            "generate_response",
            input_datas,
            num_responses=1 + num_alt_responses,
            alt_temperature_for_extra_responses=1.0 if num_alt_responses > 0 else None
        ) # List[List[dict]], List[List[str]]
    else:
        reason_actions, generated_texts = agent.predict_reason_action_batch(
            input_datas,
            num_responses=1 + num_alt_responses,
            alt_temperature_for_extra_responses=1.0 if num_alt_responses > 0 else None
        ) # List[List[dict]], List[List[str]]

    # for i in range(len(reason_actions)):
    #     for j in range(1 + num_alt_responses):
    #         print(generated_texts[i * (1 + num_alt_responses) + j])
    #         print(reason_actions[i][j])
    #         print(f"========= reason_actions {i} {j} =========")
    #         input(f"========= reason_actions {i} {j} =========")

    proposed_cars = [] # List[dict] (one per game, unlike above where there are multiple per game)
    for i in range(len(api_responses_used)):
        car_idx = reason_actions[i][0]["action"]["car_idx"]
        if car_lists[i] != [] and car_idx != 0 and car_idx <= len(car_lists[i]):
            proposed_cars.append(car_lists[i][car_idx-1])
        else:
            proposed_cars.append({})

    return reason_actions_api, generated_api_texts, api_responses, api_calls_used, api_responses_used, car_lists, reason_actions, proposed_cars, generated_texts


def rollout_batch(agent_api_call_template: Template, 
                  agent_prompt_template: Template, 
                  agent: Agent, 
                  batched_env, 
                  car_inventories: dict, 
                  batch_buyer_infos: List[dict], 
                  histories: List[List[dict]], 
                  traj_list: List[List[dict]], 
                  prev_dones: List[bool], 
                  num_alt_responses: int, 
                  past_N: int = 3,
                  all_prev_api_calls_in: List[List[dict]] = [],
                  all_prev_api_calls_have_responses_in: List[List[bool]] = [],
                  num_negotiations_in: List[int] = [],
                  num_car_proposed_in: List[int] = [],
                  prev_api_calls_in: List[dict] = [],
                  prev_api_responses_in: List[List[dict]] = [],
                  prev_proposed_cars_in: List[dict] = [],
                  ):
    """
    Rollout a batch of games.

    Parameters:
        past_N: int
            The number of previous API calls to include in the prompt.
    """
    all_prev_api_calls = [[] for _ in range(len(histories))] if all_prev_api_calls_in == [] else all_prev_api_calls_in # List[List[dict]]
    all_prev_api_calls_have_responses = [[] for _ in range(len(histories))] if all_prev_api_calls_have_responses_in == [] else all_prev_api_calls_have_responses_in # List[List[bool]]
    prev_api_calls = [{} for _ in range(len(histories))] if prev_api_calls_in == [] else prev_api_calls_in # List[dict]
    prev_api_responses = [[] for _ in range(len(histories))] if prev_api_responses_in == [] else prev_api_responses_in # List[List[dict]]
    num_negotiations = [0 for _ in range(len(histories))] if num_negotiations_in == [] else num_negotiations_in # List[int]
    prev_proposed_cars = [{} for _ in range(len(histories))] if prev_proposed_cars_in == [] else prev_proposed_cars_in # List[dict]
    num_car_proposed = [0 for _ in range(len(histories))] if num_car_proposed_in == [] else num_car_proposed_in # List[int]

    while not all(prev_dones):
        # Batched way
        start_time = time.time()
        # Initialize main data structures
        data = {
            'api_reasons': [], 'api_calls': [], 'api_responses': [], 'api_scores': [],
            'api_raw_texts': [], 'api_calls_used': [], 'api_responses_used': [], 'car_lists': [],
            'reasons': [], 'actions': [], 'proposed_cars': [], 'proposed_cars_copied_in_response': [], 'scores': [], 'raw_texts': []
        }

        if num_alt_responses > 0:
            # Initialize alternative data structures
            alt_data = {
                f'alt_{field}': [[] for _ in range(len(histories))]
                for field in ['api_reasons', 'api_calls', 'api_responses', 'api_scores', 
                             'api_raw_texts', 'reasons', 'actions', 'proposed_cars', 
                             'proposed_cars_copied_in_response', 'scores', 'raw_texts']
            }
            data.update(alt_data)

        # Query the agent
        api_reasons_actions_dict, generated_api_texts, api_responses_raw, api_calls_used_raw, api_responses_used_raw, car_lists, reasons_actions_dict, proposed_cars_raw, generated_raw_texts = query_agent_batch(agent_api_call_template, agent_prompt_template, agent, histories, all_prev_api_calls, all_prev_api_calls_have_responses, prev_api_calls, prev_api_responses, num_alt_responses, batch_buyer_infos, car_inventories, past_N)

        # Update the trajectories (Only if the game is not done)
        for i in range(len(histories)):
            if prev_dones[i]:
                # Not getting added to the trajectories
                data['api_reasons'].append("")
                data['api_calls'].append("")
                data['api_responses'].append("")
                data['api_scores'].append(None)
                data['api_raw_texts'].append("")
                data['api_calls_used'].append("")
                data['api_responses_used'].append("")
                data['car_lists'].append([])
                data['reasons'].append("")
                data['actions'].append("")
                data['proposed_cars'].append({})
                data['proposed_cars_copied_in_response'].append({})
                data['scores'].append(None)
                data['raw_texts'].append("")
            else:
                # Add the APIs to all prev api calls
                all_prev_api_calls[i].append(api_reasons_actions_dict[i][0]["action"])
                all_prev_api_calls_have_responses[i].append(api_responses_raw[i][0] != [])

                # Previous API call gets set to the API call used
                prev_api_calls[i] = api_calls_used_raw[i]
                prev_api_responses[i] = car_lists[i] # Because api responses might be truncated

                # Adding the first instance to be the taken action
                data['api_reasons'].append(api_reasons_actions_dict[i][0]["reason"])
                data['api_calls'].append(api_reasons_actions_dict[i][0]["action"])
                data['api_responses'].append(api_responses_raw[i][0])
                data['api_scores'].append(api_reasons_actions_dict[i][0]["score"] if "score" in api_reasons_actions_dict[i][0] else None)
                data['api_raw_texts'].append(generated_api_texts[i])

                data['api_calls_used'].append(api_calls_used_raw[i]) # Because api_call_used is List[dict]
                data['api_responses_used'].append(api_responses_used_raw[i]) # Because api_responses_used is List[dict]
                data['car_lists'].append(car_lists[i])

                data['reasons'].append(reasons_actions_dict[i][0]["reason"])
                data['actions'].append(reasons_actions_dict[i][0]["action"]["response"])
                data['proposed_cars'].append(proposed_cars_raw[i])
                data['proposed_cars_copied_in_response'].append(reasons_actions_dict[i][0]["action"]["proposed_car"])
                data['scores'].append(reasons_actions_dict[i][0]["score"] if "score" in reasons_actions_dict[i][0] else None)
                data['raw_texts'].append(generated_raw_texts[i*(1+num_alt_responses)])

                if num_alt_responses > 0:
                    for j in range(num_alt_responses):
                        data[f'alt_api_reasons'][i].append(api_reasons_actions_dict[i][j+1]["reason"])
                        data[f'alt_api_calls'][i].append(api_reasons_actions_dict[i][j+1]["action"])
                        data[f'alt_api_responses'][i].append(api_responses_raw[i][j+1])
                        data[f'alt_api_scores'][i].append(api_reasons_actions_dict[i][j+1]["score"] if "score" in api_reasons_actions_dict[i][j+1] else None)
                        data[f'alt_api_raw_texts'][i].append(generated_api_texts[i*(1+num_alt_responses) + j + 1])

                        data[f'alt_reasons'][i].append(reasons_actions_dict[i][j+1]["reason"])
                        data[f'alt_actions'][i].append(reasons_actions_dict[i][j+1]["action"]["response"])
                        data[f'alt_proposed_cars'][i].append(reasons_actions_dict[i][j+1]["action"]["proposed_car"])  # We use the car index instead (for simplicity)
                        data[f'alt_proposed_cars_copied_in_response'][i].append(reasons_actions_dict[i][j+1]["action"]["proposed_car"])
                        data[f'alt_scores'][i].append(reasons_actions_dict[i][j+1]["score"] if "score" in reasons_actions_dict[i][j+1] else None)
                        data[f'alt_raw_texts'][i].append(generated_raw_texts[i*(1+num_alt_responses) + j + 1])
                        
        print(f"[AGENT] time taken for batch_size={len(histories)}: {time.time() - start_time}")

        # Step the environment
        histories, buyer_reasons, buyer_responses, buyer_decisions, rewards, successes, failure_reasons, dones, num_negotiations, prev_proposed_cars, num_car_proposed = batched_env.step(batch_buyer_infos, histories, data['actions'], data['proposed_cars'], data['proposed_cars_copied_in_response'], num_negotiations, prev_proposed_cars, num_car_proposed, car_inventories, prev_dones)

        # Log the trajectories
        for i in range(len(histories)):
            if not prev_dones[i]:
                traj_list[i].append({
                    "step": len(traj_list[i]),
                    "api_reason": data['api_reasons'][i],
                    "api_call": data['api_calls'][i],
                    "api_response": data['api_responses'][i],
                    "api_score": data['api_scores'][i],
                    "api_raw_text": data['api_raw_texts'][i],
                    "api_call_used": data['api_calls_used'][i],
                    "api_response_used": data['api_responses_used'][i],
                    "reason": data['reasons'][i],
                    "action": data['actions'][i],
                    "proposed_car": data['proposed_cars'][i],
                    "raw_text": data['raw_texts'][i],
                    "buyer_reason": buyer_reasons[i],
                    "buyer_response": buyer_responses[i],
                    "buyer_decision": buyer_decisions[i],
                    "reward": rewards[i],
                    "success": successes[i],
                    "failure_reason": failure_reasons[i],
                    "score": data['scores'][i],
                    "api_alternatives": [
                        {
                            "api_reason": data[f'alt_api_reasons'][i][j],
                            "api_call": data[f'alt_api_calls'][i][j],
                            "api_response": data[f'alt_api_responses'][i][j],
                            "api_score": data[f'alt_api_scores'][i][j],
                            "api_raw_text": data[f'alt_api_raw_texts'][i][j],
                        }
                        for j in range(num_alt_responses)
                    ] if num_alt_responses > 0 else None,
                    "alternatives": [
                        {
                            "reason": data[f'alt_reasons'][i][j],
                            "action": data[f'alt_actions'][i][j],
                            "score": data[f'alt_scores'][i][j],
                            "raw_text": data[f'alt_raw_texts'][i][j]
                        }
                        for j in range(num_alt_responses)
                    ] if num_alt_responses > 0 else None
                })

            if len(traj_list[i]) == 1:
                # Also log the buyer info (for the first step)
                traj_list[i][0]["buyer_info"] = batch_buyer_infos[i]

        prev_dones = dones

    return traj_list