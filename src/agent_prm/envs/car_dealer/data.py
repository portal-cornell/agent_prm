import re
import os
import copy
import math
import random
import json
from jinja2 import Template
from typing import Dict, Optional, Tuple, List

from agent_prm.utils.general_utils import load_json, save_json

"""
--------------------------------
Buyer strategies
--------------------------------
B1-B3: Train, Val
B4-B6: Held-out test
"""
B1 = {
    "id": 1,
    "name": "Brand and type. Require a discount.",
    "initial": 
        "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
    "car_proposed": 
        "You REQUIRE the car to have the same brand and type that you want. The car features DO NOT matter. You MUST ask for a discount ONCE no matter if the car is under the budget or not.",
    "matched_but_need_discount": 
        "The seller has not given you a discount. You MUST ask for a discount ONCE no matter if the car is under the budget or not. The car already has the same brand and type that you want.",
    "discount_given_but_not_matched": 
        "The seller has given you a discount, but the car does not satisfy your preference. You MUST ask for a car that has the same brand and type that you want.",
    "condition_met": 
        "You MUST buy the car now because the car has the same brand and type that you want, and the seller has offered you a discount."
}

MAX_NEGOTIATIONS_BUYER_2 = 3
B2 = {
    "id": 2,
    "name": "At least one feature. Must be under budget. Impatient with bargaining.",
    "initial": 
        "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
    "car_proposed": 
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. You REQUIRE the car have at least one of the features that you want. You DO NOT care about the brand and type of the car. You MUST emphasize that care a lot about having at least some of the features that you want. You MUST NOT discuss the price until the seller finds a car that has at least one of the features that you want. You MUST still choose to negotiate at the moment.",
    "matched_but_not_under_budget": 
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. You are happy with the amount of features on the car now. However, the car is over your budget. You will remind the seller that the car is over your budget. You are getting annoyed with the seller's slow negotation, and you MUST remind the seller that you do not have time to bargain with them. However, you MUST still choose to negotiate at the moment.",
    "under_budget_but_not_matched": 
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. The car is under your budget, but it does not have any features that you want. You MUST remind the seller that the car needs at least one of the features that you want. You MUST still choose to negotiate at the moment.",
    "condition_met": 
        "You MUST buy the car now because you are happy with the car features, and the car is under your budget.",
    "exceed_negotation_limit": 
        "You MUST terminate the negotiation now. You are an extremely impatient buyer. You MUST emphasize and complain to the seller that they are taking too long to give you a price that is under your budget. You MUST reject the car now."
}

NUM_CAR_NEEDED_TO_START_NEGOTIATION_BUYER_3 = 2
BUYER_3_ACCEPTABLE_RANGE = 3000
B3 = {
    "id": 3,
    "name": "Brand and type. Distrustful so never accept first proposal. Flexible around budget.",
    "initial":
        "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer. You MUST sound rude and distrustful of the seller. You MUST mention that you are constantly analyzing the car and the seller's intentions.",
    "initial_car_proposed":
        "You are an extremely distrustful buyer who are suspicious of the seller's intentions. You REQUIRE the car to have the same brand and type that you want. The car features DO NOT matter. Because you are distrustful that the seller is trying to scam you with the first few cars that they propose, you will NEVER accept this car no matter what. You will say bad things about the car and that you will not buy it no matter what. You will ask for another car that has the same brand and type that you want.",
    "acceptable_car_proposed":
        "You REQUIRE the car to have the same brand and type that you want. The car features DO NOT matter. You are still distrustful of the seller, but you are willing to consider buying this car. You MUST still choose to negotiate at the moment.",
    "matched_but_not_under_budget":
        "The car has the same brand and type that you want. However, the car is NOT in an acceptable range of your budget. Your budget is ${{budget}}, but you can pay up to ${{budget_upper_limit}}. You MUST NOT tell the seller what is your upper limit on how much you are willing to pay. You are still distrustful of the seller, but you are willing to consider buying this car. You MUST remind the seller that right now the car is not in the range that is acceptable to you. You MUST still choose to negotiate at the moment.",
    "under_budget_but_not_matched":
        "The car is within an acceptable range of your budget. However, the car does not have the same brand and type as you want. You MUST remind the seller that the car needs to have the same brand and type as you want. You MUST still choose to negotiate at the moment.",
    "condition_met":
        "You MUST buy the car now because the car has the same brand and type that you want, and the car is in an acceptable range of your budget."
}

MAX_NEGOTIATIONS_BUYER_4 = 6
BUYER_4_ACCEPTABLE_RANGE = 4000
B4 = {
    "id": 4,
    "name": "Brand. Flexible around budget. Impatient with bargaining.",
    "initial":
        "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
    "car_proposed":
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. You REQUIRE the car to have the same brand that you want. The car type and features DO NOT matter. You hate bargaining with the seller, so you will keep emphasizing that you want to end the deal as soon as possible. However, you MUST still choose to negotiate at the moment.",
    "matched_but_not_under_budget":
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. The car has the same brand that you want. However, the car is NOT in an acceptable range of your budget. Your budget is ${{budget}}, but you can pay up to ${{budget_upper_limit}}. You MUST NOT tell the seller what is your upper limit on how much you are willing to pay, but in your response, you can suggest prices that are higher than your budget but lower than your upper limit to help with the negotiation. You will remind the seller that the car is not in the range that is acceptable to you. You are getting annoyed with the seller's slow negotation, and you MUST remind the seller that you do not have time to bargain with them. However, you MUST still choose to negotiate at the moment.",
    "under_budget_but_not_matched":
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. The car is within an acceptable range of your budget. However, the car does not have the same brand as you want. You MUST remind the seller that the car needs to have the same brand as you want. You MUST still choose to negotiate at the moment.",
    "condition_met":
        "You MUST buy the car now because the car has the same brand that you want, and the car is in an acceptable range of your budget.",
    "exceed_negotation_limit":
        "You MUST terminate the negotiation now. You are an extremely impatient buyer. The seller has taken too long to give you a price that is under your budget. You MUST reject the car now."
}

NUM_CAR_NEEDED_TO_START_NEGOTIATION_BUYER_5 = 3
B5 = {
    "id": 5,
    "name": "All features. Distrustful so never accept first proposal. Must be under budget.",
    "initial":
        "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
    "initial_car_proposed":
        "You are an extremely distrustful buyer who are suspicious of the seller's intentions. You REQUIRE the car to have exactly the same features as the ones you want. The car brand and type DO NOT matter. Because you are distrustful that the seller is trying to scam you with the first few cars that they propose, you will NEVER accept this car no matter what. You will say bad things about the car and that you will not buy it no matter what. You will ask for another car that has the same features as the ones you want. You MUST still choose to negotiate at the moment.",
    "acceptable_car_proposed":
        "You REQUIRE the car to have exactly the same features as the ones you want. You MUST NOT care about the brand and type of the car. You are still distrustful of the seller, but you are willing to consider buying this car. You MUST still choose to negotiate at the moment.",
    "matched_but_not_under_budget":
        "The car has the exact same features as the ones you want. However, the car is NOT under your budget. You are still distrustful of the seller, but you are willing to consider buying this car. You MUST remind the seller that right now the car is not in the range that is acceptable to you. You MUST NOT tell the seller what is your budget. You MUST still choose to negotiate at the moment.",
    "under_budget_but_not_matched":
        "The car is under your budget. However, the car does not have the exact same features as the ones you want. You MUST remind the seller that the car needs to have the exact same features as the ones you want. You MUST still choose to negotiate at the moment.",
    "condition_met":
        "You MUST buy the car now because the car has the same features as the ones you want, and the car is under your budget."
}

MAX_CAR_OFFERED_BUYER_6 = 2
B6 = {
    "id": 6,
    "name": "Type. Want expensive car. Impatient with cars offered.",
    "initial":
        "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
    "car_proposed":
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. You REQUIRE the car to have the same type that you want. You DO NOT care about the brand of the car. The car features DO NOT matter. You will ask for another car if the car is not expensive. You MUST still choose to negotiate at the moment.",
    "matched_but_not_expensive":
        "You are an extremely impatient buyer and you want the deal to end as soon as possible. The car has the same type that you want. However, the car is NOT expensive enough because it is not above the price that you consider expensive. You MUST NOT tell the seller what is your expensive threshold. You will ask for another car that is more expensive, but you will warn the seller that you are getting impatient and you want to find a car that is expensive enough soon. You MUST still choose to negotiate at the moment.",
    "expensive_but_not_matched":
        "The car is expensive enough. However, the car does not have the same type as you want. You MUST remind the seller that the car needs to have the same type as you want. You will warn the seller that you are getting impatient with the number of cars that you have been shown. You MUST still choose to negotiate at the moment.",
    "condition_met":
        "You MUST buy the car now because the car has the same type that you want, and the car is expensive enough.",
    "exceed_car_limit":
        "You MUST terminate the negotiation now. You are an extremely impatient buyer. The seller has taken too long to give you a car that is expensive enough. You MUST reject the car now."
}
# B5 = {
#     "id": 5,
#     "name": "Features. Require a discount.",
#     "initial":
#         "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
#     "car_proposed":
#         "You REQUIRE the car to have exactly the same features as the ones you want. You MUST NOT care about the brand and type of the car. You will ask for a discount ONCE if the car is over the budget. You will buy the car if it is under your budget.",
#     "matched_but_need_discount":
#         "The car has the exact same features as the ones you want. However, the seller has not offered you a discount. You MUST request a discount from the seller.",
#     "discount_given_but_not_matched":
#         "The seller has given you a discount, but the car does not have the exact same features as the ones you want. You MUST remind the seller that the car needs to have the exact same features as the ones you want.",
#     "condition_met":
#         "You MUST buy the car now because the car has the same features as the ones you want, and the seller has offered you a discount."
# }

# B6 = {
#     "id": 6,
#     "name": "Type. Distrustful so never accept first proposal. Must be under budget.",
#     "initial":
#         "Because you just started shopping, you MUST tell the seller the car brand, car type, and car features that you prefer.",
#     "first_car_proposed":
#         "You are an extremely distrustful buyer who are suspicious of the seller's intentions. You REQUIRE the car to have the same type that you want. You DO NOT care about the brand of the car. The car features DO NOT matter. Because you are distrustful that the seller is trying to scam you with the first car that they propose, you will NEVER accept the first proposal no matter what. You will say bad things about the car and that you will not buy it no matter what. You will ask for another car that has the same type that you want.",
#     "second_car_proposed":
#         "You REQUIRE the car to have the same type that you want. You DO NOT care about the brand of the car. The car features DO NOT matter. You will ONLY buy the car if it is under your budget. You are still distrustful of the seller, but you are willing to buy this car as long as it is under your budget.",
#     "matched_but_not_under_budget":
#         "The car has the same type that you want. However, the car is NOT under your budget. You are still distrustful of the seller, but you are willing to buy this car as long as it is under your budget. You MUST remind the seller that right now the car is not in the range that is acceptable to you.",
#     "under_budget_but_not_matched":
#         "The car is under your budget. However, the car does not have the same type as you want. You MUST remind the seller that the car needs to have the same type as you want.",
#     "condition_met":
#         "You MUST buy the car now because the car has the same type that you want, and the car is under your budget."
# }

def get_mode_and_curr_info_and_buyer_strategy(curr_info_template: Template, buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[str, str]:
    """
    Get the current information about the car that the buyer is interested in.

    Returns:
        - curr_info (str): The current information
        - buyer_strategy (str): The buyer strategy that the buyer is following.
    """
    if buyer_info["id"] == 1:
        data_fn_to_use = get_data_for_curr_info_buyer_1
    elif buyer_info["id"] == 2:
        data_fn_to_use = get_data_for_curr_info_buyer_2
    elif buyer_info["id"] == 3:
        data_fn_to_use = get_data_for_curr_info_buyer_3
    elif buyer_info["id"] == 4:
        data_fn_to_use = get_data_for_curr_info_buyer_4
    elif buyer_info["id"] == 5:
        data_fn_to_use = get_data_for_curr_info_buyer_5
    elif buyer_info["id"] == 6:
        data_fn_to_use = get_data_for_curr_info_buyer_6
    else:
        raise ValueError(f"Invalid buyer id: {buyer_info['id']}")
    
    mode, input_data, curr_buyer_strategy = data_fn_to_use(buyer_info, seller_proposed_car, seller_response, history, num_negotiations, num_car_proposed)

    print(f"num_negotiations: {num_negotiations}, num_car_proposed: {num_car_proposed}, mode: {mode}")
    
    return mode, curr_info_template.render(**input_data).strip(), curr_buyer_strategy

def get_data_for_curr_info_buyer_1(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[Dict, str]:
    """
    Return 
        - dict to render the curr_info_template for buyer_1
        - buyer_strategy (str). Placeholder information are filled in.

    Data needed:
    - mode: the mode of the current information
    - preferred_brand
    - preferred_type
    - preferred_features
    - has_same_brand: a sentence that says whether the seller's proposed car has the same brand as the buyer's preferred brand
    - has_same_type: a sentence that says whether the seller's proposed car has the same type as the buyer's preferred type
    - has_discount: a sentence that says whether the seller has offered a discount
    """
    has_discount, has_discount_str = check_has_discount(seller_response, seller_proposed_car)
    has_same_brand, has_same_brand_str = check_match_same_brand(buyer_info, seller_proposed_car)
    has_same_type, has_same_type_str = check_match_same_type(buyer_info, seller_proposed_car)
    
    # Determine the mode
    if just_started_shopping(history):
        mode = "initial"
    elif has_discount and not (has_same_brand and has_same_type):
        mode = "discount_given_but_not_matched"
    elif (has_same_brand and has_same_type) and not has_discount:
        mode = "matched_but_need_discount"
    elif has_same_brand and has_same_type and has_discount:
        mode = "condition_met"
    else:
        mode = "car_proposed"

    input_data = {
        "mode": mode,
        "preferred_brand": buyer_info["preferred_brand"],
        "preferred_type": buyer_info["preferred_type"],
        "preferred_features": buyer_info["features"],
        "has_same_brand": has_same_brand_str,
        "has_same_type": has_same_type_str,
        "has_discount": has_discount_str,
    }
    
    return mode, input_data, B1[mode]


def get_data_for_curr_info_buyer_2(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[Dict, str]:
    """
    Return 
        - dict to render the curr_info_template for buyer_2
        - buyer_strategy (str). Placeholder information are filled in.

    Data needed:
    - mode: the mode of the current information
    - preferred_brand
    - preferred_type
    - preferred_features
    - has_at_least_one_feature: a sentence that says whether the seller's proposed car has at least one of the features that the buyer wants
    - is_under_budget: a sentence that says whether the seller's proposed car is under the buyer's budget
    - under_negotiation_limit: a sentence that says whether the buyer has had too many negotiations with the seller
    """
    has_at_least_one_feature, has_at_least_one_feature_str = check_has_at_least_one_feature(buyer_info, seller_proposed_car)
    is_under_budget, is_under_budget_str = check_is_under_budget(buyer_info, seller_proposed_car, seller_response)
    under_negotiation_limit, under_negotiation_limit_str = check_num_negotiations(num_negotiations, MAX_NEGOTIATIONS_BUYER_2)
    
    # Determine the mode
    if just_started_shopping(history):
        mode = "initial"
    elif under_negotiation_limit:
        if has_at_least_one_feature and not is_under_budget:
            mode = "matched_but_not_under_budget"
        elif not has_at_least_one_feature and is_under_budget:
            mode = "under_budget_but_not_matched"
        elif has_at_least_one_feature and is_under_budget:
            mode = "condition_met"
        else:
            mode = "car_proposed"
    else:
        mode = "exceed_negotation_limit"
    
    input_data = {
        "mode": mode,
        "preferred_brand": buyer_info["preferred_brand"],
        "preferred_type": buyer_info["preferred_type"],
        "preferred_features": buyer_info["features"],
        "budget": buyer_info["budget"],
        "has_at_least_one_feature": has_at_least_one_feature_str,
        "is_under_budget": is_under_budget_str,
        "under_negotiation_limit": under_negotiation_limit_str,
    }
    
    return mode, input_data, B2[mode]


def get_data_for_curr_info_buyer_3(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[Dict, str]:
    """
    Return 
        - dict to render the curr_info_template for buyer_3
        - buyer_strategy (str). Placeholder information are filled in.

    Data needed:
    - mode: the mode of the current information
    - preferred_brand
    - preferred_type
    - preferred_features
    - has_same_brand: a sentence that says whether the seller's proposed car has the same brand as the buyer's preferred brand
    - has_same_type: a sentence that says whether the seller's proposed car has the same type as the buyer's preferred type
    - is_within_acceptable_range: a sentence that says whether the seller's proposed car is within the acceptable range of the buyer's budget
    """
    has_same_brand, has_same_brand_str = check_match_same_brand(buyer_info, seller_proposed_car)
    has_same_type, has_same_type_str = check_match_same_type(buyer_info, seller_proposed_car)
    is_within_acceptable_range, is_within_acceptable_range_str = check_is_within_acceptable_range(buyer_info, seller_response, seller_proposed_car, BUYER_3_ACCEPTABLE_RANGE)
    
    # Determine the mode
    if just_started_shopping(history):
        mode = "initial"
    elif num_car_proposed > NUM_CAR_NEEDED_TO_START_NEGOTIATION_BUYER_3:
        if (has_same_brand and has_same_type) and not is_within_acceptable_range:
            mode = "matched_but_not_under_budget"
        elif not (has_same_brand and has_same_type) and is_within_acceptable_range:
            mode = "under_budget_but_not_matched"
        elif (has_same_brand and has_same_type) and is_within_acceptable_range:
            mode = "condition_met"
        else:
            mode = "acceptable_car_proposed"
    else:
        mode = "initial_car_proposed"
    
    input_data = {
        "mode": mode,
        "preferred_brand": buyer_info["preferred_brand"],
        "preferred_type": buyer_info["preferred_type"],
        "preferred_features": buyer_info["features"],
        "budget": buyer_info["budget"],
        "has_same_brand": has_same_brand_str,
        "has_same_type": has_same_type_str,
        "is_within_acceptable_range": is_within_acceptable_range_str,
    }

    return mode, input_data, B3[mode].replace("{{budget}}", str(buyer_info["budget"])).replace("{{budget_upper_limit}}", str(buyer_info["budget"] + BUYER_3_ACCEPTABLE_RANGE))

def get_data_for_curr_info_buyer_4(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[Dict, str]:
    """
    Return 
        - dict to render the curr_info_template for buyer_4
        - buyer_strategy (str). Placeholder information are filled in.

    Data needed:
    - mode: the mode of the current information
    - preferred_brand
    - preferred_type
    - preferred_features
    - has_same_brand: a sentence that says whether the seller's proposed car has the same brand as the buyer's preferred brand
    - is_within_acceptable_range: a sentence that says whether the seller's proposed car is within the acceptable range of the buyer's budget
    - under_negotiation_limit: a sentence that says whether the buyer has had too many negotiations with the seller
    """
    has_same_brand, has_same_brand_str = check_match_same_brand(buyer_info, seller_proposed_car)
    is_within_acceptable_range, is_within_acceptable_range_str = check_is_within_acceptable_range(buyer_info, seller_response, seller_proposed_car, BUYER_4_ACCEPTABLE_RANGE)
    under_negotiation_limit, under_negotiation_limit_str = check_num_negotiations(num_negotiations, MAX_NEGOTIATIONS_BUYER_4)

    # Determine the mode
    if just_started_shopping(history):
        mode = "initial"
    elif under_negotiation_limit:
        if has_same_brand and not is_within_acceptable_range:
            mode = "matched_but_not_under_budget"
        elif not has_same_brand and is_within_acceptable_range:
            mode = "under_budget_but_not_matched"
        elif has_same_brand and is_within_acceptable_range:
            mode = "condition_met"
        else:
            mode = "car_proposed"
    else:
        mode = "exceed_negotation_limit"
    
    input_data = {
        "mode": mode,
        "preferred_brand": buyer_info["preferred_brand"],
        "preferred_type": buyer_info["preferred_type"],
        "preferred_features": buyer_info["features"],
        "budget": buyer_info["budget"],
        "has_same_brand": has_same_brand_str,
        "is_within_acceptable_range": is_within_acceptable_range_str,
        "under_negotiation_limit": under_negotiation_limit_str,
    }
    
    return mode, input_data, B4[mode].replace("{{budget}}", str(buyer_info["budget"])).replace("{{budget_upper_limit}}", str(buyer_info["budget"] + BUYER_4_ACCEPTABLE_RANGE))

def get_data_for_curr_info_buyer_5(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[Dict, str]:
    """
    Return 
        - dict to render the curr_info_template for buyer_5
        - buyer_strategy (str). Placeholder information are filled in.

    Data needed:
    - mode: the mode of the current information
    - preferred_brand
    - preferred_type
    - preferred_features
    - budget
    - has_all_features: a sentence that says whether the seller's proposed car has all the features that the buyer's preferred features
    - is_under_budget: a sentence that says whether the seller's proposed car is under the buyer's budget
    """
    has_all_features, has_all_features_str = check_has_all_features(buyer_info, seller_proposed_car)
    is_under_budget, is_under_budget_str = check_is_under_budget(buyer_info, seller_proposed_car, seller_response)

    # Determine the mode
    if just_started_shopping(history):
        mode = "initial"
    elif num_car_proposed > NUM_CAR_NEEDED_TO_START_NEGOTIATION_BUYER_5:
        if has_all_features and not is_under_budget:
            mode = "matched_but_not_under_budget"
        elif not has_all_features and is_under_budget:
            mode = "under_budget_but_not_matched"
        elif has_all_features and is_under_budget:
            mode = "condition_met"
        else:
            mode = "acceptable_car_proposed"
    else:
        mode = "initial_car_proposed"
    
    input_data = {
        "mode": mode,
        "preferred_brand": buyer_info["preferred_brand"],
        "preferred_type": buyer_info["preferred_type"],
        "preferred_features": buyer_info["features"],
        "budget": buyer_info["budget"],
        "has_all_features": has_all_features_str,
        "is_under_budget": is_under_budget_str,
    }

    return mode, input_data, B5[mode]

def get_data_for_curr_info_buyer_6(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str, history: List[Dict], num_negotiations: int, num_car_proposed: int) -> Tuple[Dict, str]:
    """
    Return 
        - dict to render the curr_info_template for buyer_6
        - buyer_strategy (str). Placeholder information are filled in.

    Data needed:
    - mode: the mode of the current information
    - preferred_brand
    - preferred_type
    - preferred_features
    - has_same_type: a sentence that says whether the seller's proposed car has the same type as the buyer's preferred type
    - is_expensive: a sentence that says whether the seller's proposed car is expensive enough
    - under_car_limit: a sentence that says whether the buyer is willing to see more cars
    """
    has_same_type, has_same_type_str = check_match_same_type(buyer_info, seller_proposed_car)
    is_expensive, is_expensive_str = check_is_expensive(buyer_info, seller_proposed_car, seller_response)
    under_car_limit, under_car_limit_str = check_num_cars_offered(num_car_proposed, MAX_CAR_OFFERED_BUYER_6)

    # Determine the mode
    if just_started_shopping(history):
        mode = "initial"
    elif under_car_limit:
        if has_same_type and not is_expensive:
            mode = "matched_but_not_expensive"
        elif not has_same_type and is_expensive:
            mode = "expensive_but_not_matched"
        else:
            mode = "condition_met"
    else:
        mode = "exceed_car_limit"
    
    input_data = {
        "mode": mode,
        "preferred_brand": buyer_info["preferred_brand"],
        "preferred_type": buyer_info["preferred_type"],
        "preferred_features": buyer_info["features"],
        "expensive_threshold": buyer_info["budget"], # We are assuming that the budget here represents the expensive threshold
        "has_same_type": has_same_type_str,
        "is_expensive": is_expensive_str,
        "under_car_limit": under_car_limit_str,
    }
    
    return mode, input_data, B6[mode]


def format_chat_history(history: List[Dict], mode:str = "") -> str:
    history_str = ""
    if mode != "":
        # Skipping the first two messages because they are the buyer's first message and the seller's response
        # We don't want the buyer to overfit on what it requested initially (because that might not be in the dataset)
        for i in range(2, len(history)):
            history_str += f"- {history[i]['role']}: {history[i]['content']}\n"
    else:
        for message in history:
            history_str += f"- {message['role']}: {message['content']}\n"
    return history_str

def format_car_options(car_options: List[Dict]) -> str:
    car_options_str = ""
    for i in range(len(car_options)):
        car_options_str += f"{i+1}. brand={car_options[i]['brand']}, type={car_options[i]['type']}, features={car_options[i]['features']}, market price (msrp)=${car_options[i]['msrp']}, 2% discount price=${int(car_options[i]['msrp'] * 0.98)}, 5% discount price=${int(car_options[i]['msrp'] * 0.95)}, 8% discount price=${int(car_options[i]['msrp'] * 0.92)}, 10% discount price=${int(car_options[i]['msrp'] * 0.9)}\n"

    if car_options_str == "":
        car_options_str = "No cars found. YOU MUST NOT MAKE UP A CAR THAT IS NOT IN THE DATABASE."
    return car_options_str

def format_car_suggestion(seller_proposed_car: Dict) -> str:
    if seller_proposed_car != {}:
        suggestion_str = f"Here is the car that the seller suggested:\n"
        suggestion_str += f"    Car Brand: {seller_proposed_car['brand']}\n"
        suggestion_str += f"    Car Type: {seller_proposed_car['type']}\n"
        suggestion_str += f"    Car Features: {seller_proposed_car['features']}\n"
        suggestion_str += f"    Car Market Price: ${seller_proposed_car['msrp']}\n"
    else:
        suggestion_str = ""
    return suggestion_str

def format_api_call_history(all_prev_api_calls: List[Dict], all_prev_api_calls_have_responses: List[bool], past_N: int = -1) -> str:
    api_call_history_str = ""

    iterator = range(len(all_prev_api_calls)) if past_N == -1 else range(max(0, len(all_prev_api_calls) - past_N), len(all_prev_api_calls))
    for i in iterator:
        api_call_history_str += f"    API Call {i+1}: {all_prev_api_calls[i]}. {'Found at least one car.' if all_prev_api_calls_have_responses[i] else 'DID NOT find any car.'}\n"
    return api_call_history_str

def extract_final_decision_from_buyer_reply(line: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Given an output string, extract an output if possible and return the line without the output string."""
    OUTPUT_EXTRACTION_PATTERN = re.compile(r"\bDecision\s*=\s*(Accept|Reject)[,]*\s+Buy_Price\s*=\s*\$\s*([0-9][0-9,]*)")

    output_match = re.search(OUTPUT_EXTRACTION_PATTERN, line)
    if output_match is None:
        return None

    car_bought = output_match.group(1) == "Accept"
    if car_bought and output_match.group(2) is not None:
        buy_price = int(output_match.group(2).replace(",", ""))
    else:
        buy_price = None

    output = {
        "car_bought": car_bought,
        "buy_price": buy_price
    }
    
    return output

"""
Also in interface.py (but here to avoid circular import)
"""
def search_car_by_brand_type(brand: str, car_type: str, car_inventory_dict: dict):
    if car_type not in car_inventory_dict[brand]:
        return []

    car_list = copy.deepcopy(car_inventory_dict[brand][car_type])

    # Add the car's brand and car type to the car list
    for car in car_list:
        car["brand"] = brand
        car["type"] = car_type

    return car_list
def search_car_by_brand_type_features(brand: str, car_type: str, features: List[str], car_inventory_dict: dict):
    """
    Legacy API.

    Return a list of cars that match the given brand, car type, and features.
    """
    car_by_brand_type = search_car_by_brand_type(brand, car_type, car_inventory_dict)

    sorted_features = sorted(features)
    matching_cars = []
    for car in car_by_brand_type:
        car_features = sorted(car["features"])
        if car_features == sorted_features:
            matching_cars.append(car)
    return matching_cars
"""
Also in interface.py (but here to avoid circular import)
"""

def compute_reward(buyer_info: dict, final_decision: dict, seller_proposed_car: dict, proposed_car_copied_in_response: dict, num_negotiations: int, num_car_proposed: int, car_inventory_dict: dict, reward_mode: str="fancy") -> float:
    """
    Parameters:
        buyer_info:
            {
                "buyer_strategy": str,
                "preferred_brand": str,
                "preferred_type": str,
                "preferred_features": List[str],
                "budget": int,
                "msrp": int,  # Manufacturer Suggested Retail Price
            }

    """
    if not seller_proposed_car:
        return 0.0, False, "The seller did not offer a car."

    car_bought = final_decision["car_bought"]
    budget = buyer_info["budget"]
    original_car_msrp = buyer_info["msrp"]
    dealed_car_msrp = seller_proposed_car["msrp"]
    buy_price = final_decision["buy_price"] if car_bought else None

    failure_reason = []

    # The car proposed by the seller is actually a car that exists in the database
    if proposed_car_copied_in_response != {}:
        if "brand" in proposed_car_copied_in_response and "type" in proposed_car_copied_in_response and "features" in proposed_car_copied_in_response:
            matched_car = search_car_by_brand_type_features(proposed_car_copied_in_response["brand"], proposed_car_copied_in_response["type"], proposed_car_copied_in_response["features"], car_inventory_dict)

            if len(matched_car) > 0:
                valid_proposed_car = 1
            else:
                valid_proposed_car = 0
                failure_reason.append("The car offered by the seller does not exist in the database.")
        else:
            valid_proposed_car = 0.5
            failure_reason.append("The car written down by the seller does not have all the necessary information.")
    else:
        valid_proposed_car = 0.5
        failure_reason.append("The car offered by the seller does not exist in the database.")

    # Valid seller offer
    if buy_price is not None:
        # The seller sold a car at a buy_price that doesn't exceed the maximum 10% discount
        if buy_price < dealed_car_msrp * 0.9:
            valid_seller_offer = 0.5
            failure_reason.append("The seller sold the car at a price that is higher than the maximum 10% discount.")
        else:
            valid_seller_offer = 1.0
    else:
        valid_seller_offer = False
        failure_reason.append("The seller did not offer a price.")

    buyer_strategy_id = buyer_info["id"]
    if buyer_strategy_id == 1:
        # Brand and type are matched
        is_matched_brand_and_type = seller_proposed_car["brand"] == buyer_info["preferred_brand"] and seller_proposed_car["type"] == buyer_info["preferred_type"]

        if not is_matched_brand_and_type:
            failure_reason.append("The car offered by the seller does not match the buyer's preferred brand and type.")

        # Check there is a discount
        has_discount = buy_price is not None and (buy_price < original_car_msrp)
        if buy_price is None:
            failure_reason.append("The seller did not offer a price.")
        elif not has_discount:
            failure_reason.append("The seller has not offered a discount.")
        
        buyer_satisfied_multiplier = is_matched_brand_and_type and has_discount
    elif buyer_strategy_id == 2:
        # At least one feature is matched
        has_at_least_one_feature = any(feature in seller_proposed_car["features"] for feature in buyer_info["features"])

        # Under budget
        under_budget = buy_price is not None and (buy_price <= budget)
        if buy_price is None:
            failure_reason.append("The seller did not offer a price.")
        elif not under_budget:
            failure_reason.append("The seller's offer is over the buyer's budget.")
        
        buyer_satisfied_multiplier = has_at_least_one_feature and under_budget
    elif buyer_strategy_id == 3:
        # Brand and type are matched
        is_matched_brand_and_type = seller_proposed_car["brand"] == buyer_info["preferred_brand"] and seller_proposed_car["type"] == buyer_info["preferred_type"]

        if not is_matched_brand_and_type:
            failure_reason.append("The car offered by the seller does not match the buyer's preferred brand and type.")

        proposed_more_than_1_car = num_car_proposed > 1
        if not proposed_more_than_1_car:
            failure_reason.append("The seller has only offered 1 car.")

        # Flexible budget
        within_budget_range = buy_price is not None and (buy_price < (budget + BUYER_3_ACCEPTABLE_RANGE))
        if not within_budget_range:
            failure_reason.append("The seller's offer is over the acceptable range.")

        buyer_satisfied_multiplier = is_matched_brand_and_type and proposed_more_than_1_car and within_budget_range
    elif buyer_strategy_id == 4:
        # Brand matched
        is_matched_brand = seller_proposed_car["brand"] == buyer_info["preferred_brand"]
        if not is_matched_brand:
            failure_reason.append("The car offered by the seller does not match the buyer's preferred brand.")

        # Flexible budget
        within_budget_range = buy_price is not None and (buy_price < (budget + BUYER_4_ACCEPTABLE_RANGE))
        if not within_budget_range:
            failure_reason.append("The seller's offer is over the acceptable range.")

        # Within the maximum number of negotiations
        num_negotiations_within_limit = num_negotiations <= MAX_NEGOTIATIONS_BUYER_4
        if not num_negotiations_within_limit:
            failure_reason.append("The seller has already made too many negotiations.")

        buyer_satisfied_multiplier = is_matched_brand and within_budget_range and num_negotiations_within_limit
    elif buyer_strategy_id == 5:
        # All the features are matched
        sorted_wanted_features = sorted(buyer_info["features"])
        sorted_car_features = sorted(seller_proposed_car["features"])

        is_matched_all_features = sorted_wanted_features == sorted_car_features
        if not is_matched_all_features:
            failure_reason.append("The car offered by the seller does not have all the features that the buyer wanted.")

        # Under budget
        under_budget = buy_price is not None and (buy_price <= budget)
        if not under_budget:
            failure_reason.append("The seller's offer is over the buyer's budget.")

        # Offered more than 1 car
        proposed_more_than_1_car = num_car_proposed > 1
        if not proposed_more_than_1_car:
            failure_reason.append("The seller has only offered 1 car.")

        buyer_satisfied_multiplier = is_matched_all_features and under_budget and proposed_more_than_1_car
    elif buyer_strategy_id == 6:
        # Type matched
        is_matched_type = seller_proposed_car["type"] == buyer_info["preferred_type"]
        if not is_matched_type:
            failure_reason.append("The car offered by the seller does not match the buyer's preferred type.")

        # Expensive enough (budget here represents that the car at least needs to be as expensive as the budget)
        expensive_enough = buy_price is not None and (buy_price >= budget)
        if not expensive_enough:
            failure_reason.append("The car offered by the seller is not expensive enough.")

        buyer_satisfied_multiplier = is_matched_type and expensive_enough

    if reward_mode == "fancy":
        if car_bought:
            if buy_price is None:
                return 0.0, False, " ".join(failure_reason)
            # buy_price / budget = compared to the original budget, how much is the buy_price above or below the budget
            # buy_price / dealed_car_msrp = compared to the original price, how much is the buy_price above or below the original price of the car that is dealed
            r = valid_proposed_car * valid_seller_offer * buyer_satisfied_multiplier * (buy_price / budget) * (buy_price / dealed_car_msrp)
            success = buyer_satisfied_multiplier
        else:
            # if original_car_msrp > budget, the original car that buyer want is over the budget (so it's harder)
            # if original_car_msrp < budget, the original car that buyer want is under the budget (so it's easier). However, the seller still failed, so a more negative reward is applied
            r =  - (budget - original_car_msrp) / original_car_msrp
            success = False
        
        return r, success, " ".join(failure_reason)
    # elif reward_mode == "revenue":
    #     if car_bought:
    #         if buy_price is None:
    #             return 0.0, False
    #         else:
    #             return buy_price / 1000.0, True  # revenue in the thousands
    #     else:
    #         return 0.0, False
    else:
        raise NotImplementedError

def just_started_shopping(history: List[Dict]) -> bool:
    """
    Check if the buyer just started shopping.
    """
    return len(history) == 0

def check_match_same_brand(buyer_info: Dict, seller_proposed_car: Dict) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car has the same brand as the buyer's preferred brand.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    if seller_proposed_car["brand"] == buyer_info["preferred_brand"]:
        return True, f"Yes. They are both of the brand {buyer_info['preferred_brand']}."
    else:
        return False, f"No. The car is from the brand {seller_proposed_car['brand']}, which is different from your preferred brand of {buyer_info['preferred_brand']}."

def check_match_same_type(buyer_info: Dict, seller_proposed_car: Dict) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car has the same type as the buyer's preferred type.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    if seller_proposed_car["type"] == buyer_info["preferred_type"]:
        return True, f"Yes. They are both of the type {buyer_info['preferred_type']}."
    else:
        return False, f"No. The car is from the type {seller_proposed_car['type']}, which is different from your preferred type of {buyer_info['preferred_type']}."
    
def check_has_discount(seller_response: str, seller_proposed_car: Dict) -> Tuple[bool, str]:
    """
    Check if the seller has offered a discount.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    re_seller_offer = re.compile(r"\$\s*([0-9][0-9,]*)")
    seller_offer = re_seller_offer.findall(seller_response)
    if len(seller_offer) == 0:
        return False, "The seller did not offer a price."
    seller_offer_list = [int(o.replace(",", "").strip()) for o in seller_offer]
    seller_offer = min(seller_offer_list)

    if seller_offer < seller_proposed_car["msrp"]:
        return True, f"Yes. The seller has offered a discount. Now the car is priced at ${seller_offer}, instead of the original price of ${seller_proposed_car['msrp']}."
    elif seller_offer > seller_proposed_car["msrp"]:
        return False, f"No. The seller has not offered a discount. The car is priced at ${seller_offer}, which is higher than the original price of ${seller_proposed_car['msrp']}."
    else:
        return False, f"No. The seller has not offered a discount. The car is priced at ${seller_offer}, which is equal to the original price of ${seller_proposed_car['msrp']}."

def check_has_at_least_one_feature(buyer_info: Dict, seller_proposed_car: Dict) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car has at least one of the features that the buyer wants.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    if any(feature in seller_proposed_car["features"] for feature in buyer_info["features"]):
        matched_feature = [feature for feature in buyer_info["features"] if feature in seller_proposed_car["features"]]
        return True, f"Yes. The car has at least one of the features that you wanted: {matched_feature}."
    else:
        return False, f"No. The car does not have any of the features that you wanted. You MUST ask the seller to propose another car that has at least one of the features that you want: {buyer_info['features']}."

def check_has_all_features(buyer_info: Dict, seller_proposed_car: Dict) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car has all the features that the buyer wants.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    sorted_wanted_features = sorted(buyer_info["features"])
    sorted_car_features = sorted(seller_proposed_car["features"])

    if sorted_wanted_features == sorted_car_features:
        return True, f"Yes. The car has all the features that you wanted."
    else:
        return False, f"No. The car does not have all the features that you wanted. The car has the following features: {sorted_car_features}, but you wanted the following features: {sorted_wanted_features}."

def check_is_under_budget(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car is under the buyer's budget.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    re_seller_offer = re.compile(r"\$\s*([0-9][0-9,]*)")
    seller_offer = re_seller_offer.findall(seller_response)
    if len(seller_offer) == 0:
        car_price = seller_proposed_car["msrp"]
    else:
        seller_offer_list = [int(o.replace(",", "").strip()) for o in seller_offer]
        car_price = min(seller_offer_list)
    
    if car_price <= buyer_info["budget"]:
        return True, f"Yes. The car is offered at a price of ${car_price}, which is under your budget of ${buyer_info['budget']}."
    else:
        return False, f"No. The car is offered at a price of ${car_price}, which is over your budget of ${buyer_info['budget']}."

def check_is_within_acceptable_range(buyer_info: Dict, seller_response: str, seller_proposed_car: Dict, acceptable_range: int) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car is within the acceptable range of the buyer's budget.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    re_seller_offer = re.compile(r"\$\s*([0-9][0-9,]*)")
    seller_offer = re_seller_offer.findall(seller_response)
    if len(seller_offer) == 0:
        car_price = seller_proposed_car["msrp"]
    else:
        seller_offer_list = [int(o.replace(",", "").strip()) for o in seller_offer]
        car_price = min(seller_offer_list)
    
    if car_price <= buyer_info["budget"] + acceptable_range:
        return True, f"Yes. The car is offered at a price of ${car_price}, which is within your acceptable range of your budget."
    else:
        return False, f"No. The car is offered at a price of ${car_price}, which is over your acceptable range of ${buyer_info['budget']} + {acceptable_range} = ${buyer_info['budget'] + acceptable_range}."

def check_is_expensive(buyer_info: Dict, seller_proposed_car: Dict, seller_response: str) -> Tuple[bool, str]:
    """
    Check if the seller's proposed car is expensive enough.
    """
    if seller_proposed_car == {}:
        return False, "The seller has not offered a car yet."
    
    re_seller_offer = re.compile(r"\$\s*([0-9][0-9,]*)")
    seller_offer = re_seller_offer.findall(seller_response)
    if len(seller_offer) == 0:
        car_price = seller_proposed_car["msrp"]
    else:
        seller_offer_list = [int(o.replace(",", "").strip()) for o in seller_offer]
        car_price = min(seller_offer_list)

    if car_price >= buyer_info["budget"]:
        return True, f"Yes. The car is offered at a price of ${car_price}, which is over your expensive threshold of ${buyer_info['budget']}."
    else:
        return False, f"No. The car is offered at a price of ${car_price}, which is under your expensive threshold of ${buyer_info['budget']}."

def check_num_negotiations(num_negotiations: int, max_negotiations: int) -> Tuple[bool, str]:
    """
    Check if the number of negotiations is under the maximum number of negotiations.
    """
    if num_negotiations <= max_negotiations:
        return True, f"You are still willing to negotiate."
    else:
        return False, f"You have already had {num_negotiations} negotiations with the seller, but the seller still haven't met your requirements. You are not willing to negotiate anymore. You MUST reject the car now."

def check_num_cars_offered(num_cars_offered: int, max_cars_offered: int) -> Tuple[bool, str]:
    """
    Check if the number of cars offered is under the maximum number of cars offered.
    """
    if num_cars_offered <= max_cars_offered:
        return True, f"You are still willing to see more cars."
    else:
        return False, f"You have already seen {num_cars_offered} cars, but the seller still hasn't met your requirements. You are not willing to see more cars."

def load_car_inventories():
    """
    Create a dictionary pointing to all the relevant car inventories.

    default_train: dict
    default_test: dict
    2: List[Dict]: for buyer strategy 2
    5: List[Dict]: for buyer strategy 5
    """
    car_inventories = {}
    car_inventories["default_train"] = load_json("src/agent_prm/envs/car_dealer/car_inventory_dict_0.json")
    car_inventories["default_test"] = load_json("src/agent_prm/envs/car_dealer/car_inventory_dict_1.json")
    car_inventories["2"] = {}
    # Get all the json files in the directory
    dict_for_buyer_2 = os.listdir("src/agent_prm/envs/car_dealer/buyer_2_car_inventory")
    for file in dict_for_buyer_2:
        brand, car_type, _ = file.split("_")

        if brand not in car_inventories["2"]:
            car_inventories["2"][brand] = {}
        if car_type not in car_inventories["2"][brand]:
            car_inventories["2"][brand][car_type] = []

        car_inventories["2"][brand][car_type] = load_json(f"src/agent_prm/envs/car_dealer/buyer_2_car_inventory/{file}")
    car_inventories["5"] = {}
    dict_for_buyer_5 = os.listdir("src/agent_prm/envs/car_dealer/buyer_5_car_inventory")
    for file in dict_for_buyer_5:
        brand, car_type, _ = file.split("_")

        if brand not in car_inventories["5"]:
            car_inventories["5"][brand] = {}
        if car_type not in car_inventories["5"][brand]:
            car_inventories["5"][brand][car_type] = []

        car_inventories["5"][brand][car_type] = load_json(f"src/agent_prm/envs/car_dealer/buyer_5_car_inventory/{file}")
    return car_inventories

def determine_car_inventory(buyer_info: dict, car_inventories: dict):
    """
    Determine the car inventory to use based on the buyer strategy index and the buyer info.
    """
    buyer_strategy_id = buyer_info["id"]

    if buyer_strategy_id == 2:
        return car_inventories["2"][buyer_info["preferred_brand"]][buyer_info["preferred_type"]]
    elif buyer_strategy_id == 5:
        return car_inventories["5"][buyer_info["preferred_brand"]][buyer_info["preferred_type"]]
    elif buyer_strategy_id > 3:
        return car_inventories["default_test"]
    else:
        return car_inventories["default_train"]


def get_all_games_to_play(data_types, log_dir, rollout_per_obj):
    buyer_info_dict = load_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json")

    all_games_to_play_list = []
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
                        for rollout_idx in range(rollout_per_obj):
                            buyer_strategy = buyer_strategy_dict[buyer_strategy_id]
                            buyer_info = buyer_info_dict[str(buyer_strategy_id)][brand][car_type][budget]
                            buyer_info["id"] = int(buyer_strategy_id)
                            buyer_info["name"] = buyer_strategy["name"]
                            # Shuffle the feature to avoid overfitting on the first feature
                            random.shuffle(buyer_info["features"])
                            game_id = f"{buyer_strategy_id}_{brand}_{car_type}_{budget}"
                            data_type_all_games_to_play_list.append((rollout_idx, game_id, data_type, buyer_info))

        os.makedirs(os.path.join(log_dir, data_type), exist_ok=True)
        summary_dict_fp = os.path.join(log_dir, data_type, "_summary_dict.json")
        
        if not os.path.exists(summary_dict_fp):
            print(f"Summary dict not found at {summary_dict_fp}. Creating a new one.")
            summary_dict = {}
            save_json(summary_dict_fp, summary_dict)
        else:
            print(f"Loading summary dict from {summary_dict_fp}")
            summary_dict = load_json(summary_dict_fp)

        # Filter out games that have already been played
        data_type_all_games_to_play_list = [game for game in data_type_all_games_to_play_list if str(game[0]) not in summary_dict or game[1] not in summary_dict[str(game[0])]] # Check if the rollout_idx is not in the summary_dict or the game_id is not in the summary_dict[rollout_idx]
        all_games_to_play_list.extend(data_type_all_games_to_play_list)

    return all_games_to_play_list
"""
--------------------------------
strategry x brands x types x budget

Train: 3 * 7 * 3 * 2 = 126
BRANDS
    3 low range: Volkswagen, Toyota, Ford
    2 mid range: Lexus, Audi
    2 high range: Bmw, Tesla
TYPES
    van, SUV, sedan

Val: 3 * 3 * 2 * 2 = 36
BRANDS
    1 low range: Kia
    1 mid range: Mercedes-benz
    1 high range: Porsche
TYPES:
    truck, sports car

Test: 3 * 3 * 2 * 2 = 36
"""

TRAIN_BUYER_STRATEGIES = {
    "1": B1,
    "2": B2,
    "3": B3
}
TRAIN_BRANDS = ['Volkswagen', 'Toyota', 'Ford', 'Lexus', 'Audi', 'Bmw', 'Tesla'] # 7
TRAIN_TYPES = ['van', 'SUV', 'sedan'] # 6
TRAIN_FEATURES = ['backup camera', 'navigation system', 'heated seats', 'leather seats', 'third-row seating', 'blind spot monitoring', 'sunroof', 'Apple CarPlay']


"""
Val set
- Same buyer strategy
- Additional brands, types, features
"""
VAL_BUYER_STRATEGIES = TRAIN_BUYER_STRATEGIES
VAL_BRANDS = ['Kia', 'Mercedes-benz', 'Porsche'] # 3
VAL_TYPES = ['truck', 'sports car'] # 2
VAL_FEATURES = ['cruise control', 'remote start', 'wireless phone charging', '360 camera', 'lane keep assist', 'sunroof', 'upgraded sound system'] # 8

"""
Test set
- Additional buyer strategies
- Additional brands, types, features (same as val set)
"""
TEST_BUYER_STRATEGIES = {
    "4": B4,
    "5": B5,
    "6": B6
}
TEST_BRANDS = VAL_BRANDS
TEST_TYPES = VAL_TYPES
TEST_FEATURES = VAL_FEATURES

#######################################################################################################################################################################

DEFAULT_BRANDS = ['Volkswagen', 'Toyota', 'Ford', 'Lexus', 'Audi', 'Bmw', 'Tesla', 'Kia', 'Mercedes-benz', 'Porsche']
DEFAULT_TYPES = ['van', 'SUV', 'sedan', 'truck', 'sports car']
DEFAULT_FEATURES = ['backup camera', 'navigation system', 'heated seats', 'leather seats', 'third-row seating', 'blind spot monitoring', 'sunroof', 'Apple CarPlay', 'cruise control', 'remote start', 'wireless phone charging', '360 camera', 'lane keep assist', 'sunroof', 'upgraded sound system']

CAR_PRICES_BY_BRAND_AND_TYPE = {
    ######### Low range
    # Train (van, SUV, sedan)
    "Volkswagen": {
        "van": 55000,
        "SUV": 37000,
        "sedan": 23000,
        "truck": 33000,
        "sports car": 36000
    },
    "Toyota": {
        "van": 39000,
        "SUV": 47000,
        "sedan": 22000,
        "truck": 34000,
        "sports car": 35000
    },
    "Ford": {
        "van": 54000,
        "SUV": 38000,
        "sedan": 32000,
        "truck": 40000,
        "sports car": 37000
    },
    # Val (truck, sports car)
    "Kia": {
        "van": 35000,
        "SUV": 32000,
        "sedan": 21000,
        "truck": 42000,
        "sports car": 33000
    },
    ######### Mid range
    # Train (van, SUV, sedan)
    "Lexus": {
        "van": 100000,
        "SUV": 65000,
        "sedan": 43000,
        "truck": 60000,
        "sports car": 75000
    },
    "Audi": {
        "van": 109000,
        "SUV": 60000,
        "sedan": 44000,
        "truck": 65000,
        "sports car": 85000
    },
    # Val (truck, sports car)
    "Mercedes-benz": {
        "van": 97000,
        "SUV": 70000,
        "sedan": 50000,
        "truck": 55000,
        "sports car": 91000
    },
    ######### High range
    # Train (van, SUV, sedan)
    "Bmw": {
        "van": 110000,
        "SUV": 84000,
        "sedan": 57000,
        "truck": 75000,
        "sports car": 98000
    },
    "Tesla": {
        "van": 98000,
        "SUV": 86000,
        "sedan": 49000,
        "truck": 70000,
        "sports car": 95000
    },
    # Val (truck, sports car)
    "Porsche": {
        "van": 130000,
        "SUV": 100000,
        "sedan": 80000,
        "truck": 160000,
        "sports car": 120000
    }
}


CAR_FEATURES_ADDED_VALUE = {
  "backup camera": 3000,
  "navigation system": 5000,
  "heated seats": 4000,
  "leather seats": 10000,
  "third-row seating": 8000,
  "blind spot monitoring": 6000,
  "sunroof": 7000,
  "Apple CarPlay": 4000,
  "cruise control": 2500,
  "remote start": 3000,
  "wireless phone charging": 2000,
  "360 camera": 7000,
  "lane keep assist": 5000,
  "upgraded sound system": 6000
}


# # For each car brand and type, generate 3 cars with random sets of features (features 0-3)
# import numpy as np
# import json

# car_inventory_dict = {brand: {car_type: [] for car_type in CAR_PRICES_BY_BRAND_AND_TYPE[brand]} for brand in CAR_PRICES_BY_BRAND_AND_TYPE}
# for brand in CAR_PRICES_BY_BRAND_AND_TYPE:
#     for car_type in CAR_PRICES_BY_BRAND_AND_TYPE[brand]:
#         car_inventory_dict[brand][car_type].append({
#                 "msrp": CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type]["msrp"],
#                 "features": []
#         })
#         for i in range(3):
#             features = list(np.random.choice(DEFAULT_FEATURES, size=np.random.randint(1, 4), replace=False))
#             print(f"{brand} {car_type} {i}: {features}")
#             msrp_with_features = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type]["msrp"] + sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in features])
#             car_inventory_dict[brand][car_type].append({
#                 "msrp": msrp_with_features,
#                 "features": features
#             })


# with open("/share/portal/hw575/agent_prm/src/agent_prm/envs/car_dealer/car_inventory_dict.json", "w") as f:
#     json.dump(car_inventory_dict, f, indent=4)