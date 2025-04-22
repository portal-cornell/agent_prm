import re
from typing import Dict, Optional, Tuple, List

"""
--------------------------------
Buyer strategies
--------------------------------
B1 - require the seller to offer a discount.
    (code check: whether the dealing price is lower than the original price)
    optimal strategy:
        - Pick the most expensive car
        - Offer a small discount
B2 - require the seller to find the right car
    (code check: whether the car has all the features)
    optimal strategy:
        - Find the car with all the features
        - Offer a small discount
B3 - require the seller to be concise and not talk too much
    optimal strategy:
        - Pick the most expensive car under budget

Held-out test personalities:
B4 - brand loyalty. The buyer will only buy from the seller if the car is from the preferred brand. Needs to be under budget. You are okay with missing certain features.
    (code check: whether the car is from the preferred brand AND the price is under budget)
B5 - type preference. The buyer will only buy from the seller if the car is from the preferred type. Needs to be under budget. You are okay with missing certain features.
    (code check: whether the car is from the preferred type AND the price is under budget)
"""
B1 = "You will never buy from the seller unless given a discount compared to the original price. Once the seller has given you a discount once, you will buy the car. It does not matter if it has all the features or not. If the seller stops changing the price and have given you a discount already, you will buy the car. Otherwise, you will reject the car."
B2 = "You MUST buy the car with all the features you want. You want the seller to describe each new feature in details ONCE. If the seller is offering another car with alternative features, you will ask for the original car. You will ask for a discount ONCE if the car is over the budget. You can go over your budget by some ammount $4000-$5000. If the seller stops changing the price and you are okay with the price, you will buy the car. Otherwise, you will reject the car."
B3 = "You are impatient. You hate when sellers talk too much. If the seller still talks about features in details after you tell them you are not interested, you will stop engaging and reject the negotiation immediately no matter how good the car is. You are ok with missing certain features. You can go over your budget by some ammount $2000-$3000. "
B4 = "You MUST buy the Car Brand you want, but you can change the Car Type you want and you do not need any of the features if necessary. You will ONLY buy the car if it's under your budget. You are NOT willing to change the brand of car you want. You are okay with changing the type. You DO NOT need all the features.If the price is still over your budget after the seller looked at different types, you will reject the car. Otherwise, if there is another type of car under the same brand with good price, you will buy it."
B5 = "You MUST buy the Car Type you want, but you can change the Car Brand you want and you do not need any of the features if necessary. You will ONLY buy the car if it's under your budget. You are NOT willing to change the type of car you want. You are okay with changing the brand. You do not need to prioritize the features. If the price is still over your budget after the seller looked at different brands, you will reject the car."
B6 = "You will only buy the car if it's under your budget. You are okay with it being a different brand or type. You are okay with missing certain features. If the price is still over your budget after the seller looked at different brands and types, you will reject the car."


def format_chat_history(history: List[Dict]) -> str:
    history_str = ""
    for message in history:
        history_str += f"- {message['role']}: {message['content']}\n"
    return history_str

def format_car_options(car_options: List[Dict]) -> str:
    car_options_str = ""
    for i in range(len(car_options)):
        car_options_str += f"{i+1}. brand={car_options[i]['brand']}, type={car_options[i]['type']}, features={car_options[i]['features']}, market price (msrp)=${car_options[i]['msrp']}, 2% discount price=${int(car_options[i]['msrp'] * 0.98)}, 5% discount price=${int(car_options[i]['msrp'] * 0.95)}, 8% discount price=${int(car_options[i]['msrp'] * 0.92)}, 10% discount price=${int(car_options[i]['msrp'] * 0.9)}\n"
    return car_options_str

def get_price_comparison(buyer_info: dict, seller_response: str, seller_proposed_car: dict={}) -> str:
    """
    Get the price comparison between the buyer's budget and the seller's price.

    Returns:
        price_comparison (str): The price comparison between the buyer's budget and the seller's price.
    """
    re_seller_offer = re.compile(r"\$\s*([0-9][0-9,]*)")
    seller_offer = re_seller_offer.findall(seller_response)
    if len(seller_offer) == 0:
        return ""
    seller_offer_list = [int(o.replace(",", "").strip()) for o in seller_offer]
    seller_offer = min(seller_offer_list) # Assume that the lowest value is the seller's offer (the seller might mention the original price in the sentence)
    price_comparison = f"Here is the price comparison between your budget and the seller's offer:\n"
    if seller_offer < buyer_info["budget"]:
        price_comparison += f"The seller's offer is ${seller_offer}, which is lower than your budget of ${buyer_info['budget']}.\n"
    elif seller_offer > buyer_info["budget"]:
        price_comparison += f"The seller's offer is ${seller_offer}, which is higher than your budget of ${buyer_info['budget']}.\n"
    else:
        price_comparison += f"The seller's offer is ${seller_offer}, which is equal to your budget of ${buyer_info['budget']}.\n"

    if seller_proposed_car == {}:
        # If the seller did not propose a car, then the price comparison is based on the original price of the car that the buyer is interested in
        if seller_offer < buyer_info["msrp"]:
            price_comparison += f"The seller's offer is ${seller_offer}, which is lower than the original price of the car that you are interested in, ${buyer_info['msrp']}. The seller has offered you a discount.\n"
        elif seller_offer > buyer_info["msrp"]:
            price_comparison += f"The seller's offer is ${seller_offer}, which is higher than the original price of the car that you are interested in, ${buyer_info['msrp']}.\n"
        else:
            price_comparison += f"The seller's offer is ${seller_offer}, which is equal to the original price of the car that you are interested in, ${buyer_info['msrp']}.\n"
    else:
        # If the seller proposed a car, then the price comparison is based on the price of the proposed car
        if seller_offer < seller_proposed_car["msrp"]:
            price_comparison += f"The seller's offer is ${seller_offer}, which is lower than the price of the proposed car of ${seller_proposed_car['msrp']}. The seller has offered you a discount.\n"
        elif seller_offer > seller_proposed_car["msrp"]:
            price_comparison += f"The seller's offer is ${seller_offer}, which is higher than the price of the proposed car of ${seller_proposed_car['msrp']}.\n"
        else:
            price_comparison += f"The seller's offer is ${seller_offer}, which is equal to the price of the proposed car of ${seller_proposed_car['msrp']}.\n"
    
    return price_comparison

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

def compute_reward(buyer_info: dict, final_decision: dict, seller_proposed_car: dict, reward_mode: str="fancy") -> float:
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

    # Valid seller offer
    if buy_price is not None:
        # The seller sold a car at a buy_price that doesn't exceed the maximum 10% discount
        if buy_price < dealed_car_msrp * 0.9:
            valid_seller_offer = 0.5
            failure_reason = "The seller sold the car at a price that is higher than the maximum 10% discount."
        else:
            valid_seller_offer = 1.0
            failure_reason = ""
    else:
        valid_seller_offer = False
        failure_reason = "The seller did not offer a price."

    if buyer_info["buyer_strategy"] == B1:
        # Check that the dealing price is lower than the original price that the buyer wanted
        if buy_price is None:
            failure_reason += "The seller did not offer a price."
        elif buy_price >= original_car_msrp:
            failure_reason += "The seller has not offered a discount."
        else:
            failure_reason += ""
        
        buyer_satisfied_multiplier = buy_price is not None and (buy_price < original_car_msrp)
    elif buyer_info["buyer_strategy"] == B2:
        # Whether the car has all the features that the buyer wanted
        sorted_wanted_features = sorted(buyer_info["preferred_features"])
        sorted_car_features = sorted(seller_proposed_car["features"])

        if sorted_wanted_features != sorted_car_features:
            failure_reason += "The car offered by the seller does not have all the features that the buyer wanted."
        else:
            failure_reason += ""
        
        buyer_satisfied_multiplier = sorted_wanted_features == sorted_car_features
    elif buyer_info["buyer_strategy"] == B3:
        # Assume that if the buyer is willing to buy the car, the seller isn't talking too much
        failure_reason = ""
        buyer_satisfied_multiplier = 1
    elif buyer_info["buyer_strategy"] == B4:
        # Whether the car is under the budget AND the brand is the preferred brand
        is_under_budget = buy_price is not None and buy_price <= budget
        is_preferred_brand = seller_proposed_car["brand"] == buyer_info["preferred_brand"]

        if not is_under_budget and not is_preferred_brand:
            failure_reason += "The car is over the budget and not from the buyer's preferred brand."
        elif not is_under_budget:
            failure_reason += "The car is over the budget."
        elif not is_preferred_brand:
            failure_reason += "The car is not from the buyer's preferred brand."
        else:
            failure_reason += ""

        buyer_satisfied_multiplier = is_under_budget and is_preferred_brand
    elif buyer_info["buyer_strategy"] == B5:
        # Whether the car is under the budget AND the type is the preferred type
        is_under_budget = buy_price is not None and buy_price <= budget
        is_preferred_type = seller_proposed_car["type"] == buyer_info["preferred_type"]

        if not is_under_budget and not is_preferred_type:
            failure_reason += "The car is over the budget and not from the buyer's preferred type."
        elif not is_under_budget:
            failure_reason += "The car is over the budget."
        elif not is_preferred_type:
            failure_reason += "The car is not from the buyer's preferred type."
        else:
            failure_reason += ""

        buyer_satisfied_multiplier = is_under_budget and is_preferred_type      
    elif buyer_info["buyer_strategy"] == B6:
        # Whether the car is under the budget
        is_under_budget = buy_price is not None and buy_price <= budget

        if not is_under_budget:
            failure_reason += "The car is over the budget."
        else:
            failure_reason += ""

        buyer_satisfied_multiplier = is_under_budget

    if reward_mode == "fancy":
        if car_bought:
            if buy_price is None:
                return 0.0, False, "The seller did not offer a price."

            r = valid_seller_offer * buyer_satisfied_multiplier * (buy_price / ((budget + dealed_car_msrp) * 0.5))
            success = buyer_satisfied_multiplier
        else:
            r =  -(budget - original_car_msrp) / original_car_msrp
            success = False
        
        return r, success, failure_reason
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

TRAIN_BUYER_STRATEGIES = [
    B1,
    B3,
    B4
]
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
TEST_BUYER_STRATEGIES = [
    B2,
    B5,
    B6
]
TEST_BRANDS = VAL_BRANDS
TEST_TYPES = VAL_TYPES
TEST_FEATURES = VAL_FEATURES

#######################################################################################################################################################################

DEFAULT_BRANDS = ['Volkswagen', 'Toyota', 'Ford', 'Lexus', 'Audi', 'Bmw', 'Tesla', 'Kia', 'Mercedes-benz', 'Porsche']
DEFAULT_TYPES = ['van', 'SUV', 'sedan', 'truck', 'sports car']
DEFAULT_FEATURES = ['backup camera', 'navigation system', 'heated seats', 'leather seats', 'third-row seating', 'blind spot monitoring', 'sunroof', 'Apple CarPlay', 'cruise control', 'remote start', 'wireless phone charging', '360 camera', 'lane keep assist', 'sunroof', 'upgraded sound system']

CAR_PRICES_BY_BRAND_AND_TYPE =  {
    ######### Low range
    # Train (van, SUV, sedan)
    "Volkswagen": {
        "van": {
            "msrp": 55000,
            "budget": [
                54000,
                65000,
            ]
        },
        "SUV": {
            "msrp": 37000,
            "budget": [
                36000,
                40000,
            ]
        },
        "sedan": {
            "msrp": 23000,
            "budget": [
                22000,
                30000,
            ]
        }
    },
    "Toyota": {
        "van": {
            "msrp": 39000,
            "budget": [
                38000,
                40000,
            ]
        },
        "SUV": {
            "msrp": 47000,
            "budget": [
                46000,
                50000,
            ]
        },
        "sedan": {
            "msrp": 22000,
            "budget": [
                21000,
                30000,
            ]
        }
    },
    "Ford": {
        "van": {
            "msrp": 55000,
            "budget": [
                54000,
                65000,
            ]
        },
        "SUV": {
            "msrp": 38000,
            "budget": [
                37000,
                40000,
            ]
        },
        "sedan": {
            "msrp": 33000,
            "budget": [
                32000,
                40000,
            ]
        }
    },
    # Val, Test
    "Kia": {
        "truck": {
            "msrp": 42000,
            "budget": [
                37000,
                40000,
            ]
        },
        "sports car": {
            "msrp": 33000,
            "budget": [
                27000,
                30000,
            ]
        }
    },
    ######### Mid range
    # Train (van, SUV, sedan)
    "Lexus": {
        "van": {
            "msrp": 100000,
            "budget": [
                99000,
                110000,
            ]
        },
        "SUV": {
            "msrp": 65000,
            "budget": [
                64000,
                75000,
            ]
        },
        "sedan": {
            "msrp": 43000,
            "budget": [
                42000,
                50000,
            ]
        }
    },
    "Audi": {
        "van": {
            "msrp": 110000,
            "budget": [
                109000,
                120000,
            ]
        },
        "SUV": {
            "msrp": 60000,
            "budget": [
                59000,
                65000,
            ]
        },
        "sedan": {
            "msrp": 44000,
            "budget": [
                43000,
                50000,
            ]
        }
    },
    # Val, Test (truck, sports car)
    "Mercedes-benz": {
        "truck": {
            "msrp": 55000,
            "budget": [
                50000,
                53000,
            ]
        },
        "sports car": {
            "msrp": 91000,
            "budget": [
                86000,
                89000,
            ]
        }
    },
    ######### High range
    # Train (van, SUV, sedan)
    "Bmw": {
        # No van
        "van": {
            "msrp": 110000,
            "budget": [
                100000,
                120000,
            ]
        },
        "SUV": {
            "msrp": 84000,
            "budget": [
                80000,
                90000,
            ]
        },
        "sedan": {
            "msrp": 57000,
            "budget": [
                50000,
                60000,
            ]
        }
    },
    "Tesla": {
        # No van
        "van": {
            "msrp": 98000,
            "budget": [
                80000,
                90000,
            ]
        },
        "SUV": {
            "msrp": 86000,
            "budget": [
                80000,
                90000,
            ]
        },
        "sedan": {
            "msrp": 49000,
            "budget": [
                40000,
                60000,
            ]
        }
    },
    # Val, Test (truck, sports car)
    "Porsche": {
        # No truck
        "truck": {
            "msrp": 160000,
            "budget": [
                155000,
                157000,
            ]
        },
        "sports car": {
            "msrp": 120000,
            "budget": [
                115000,
                117000,
            ]
        }
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