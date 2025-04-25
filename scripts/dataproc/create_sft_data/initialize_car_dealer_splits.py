"""
This should only be run once to initialize the car inventory splits.

B1 (Brand and type. Require a discount): use the default car inventory. don't need any budget.
B2 (At least one feature. Must be under budget. Impatient with bargaining.): 
B3 (Brand and type. Distrustful so never accept first proposal. Flexible around budget.): 
    - use the default car inventory.
    - generate 2 budget that is reasonable within the brand and type
        the budget is between (0.9 of the cheapest car - acceptable range) and (1 of the most expensive car - acceptable range)
B4 (Brand. Flexible around budget. Impatient with bargaining.)
    - use the default car inventory.
    - generate 2 budget that is reasonable within the brand and type
        the budget is between (0.9 of the cheapest car - acceptable range) and (1 of the most expensive car - acceptable range)
B5 (All features. Distrustful so never accept first proposal. Must be under budget.):
B6 (Type. Want expensive car. Impatient with cars offered.)
    - use the default car inventory.
    - generate 2 "budget" that is within 0.8 of the range of car prices
"""
import numpy as np
import math
import json
import os
from typing import List

from agent_prm.envs.car_dealer.data import CAR_PRICES_BY_BRAND_AND_TYPE, TRAIN_FEATURES, DEFAULT_FEATURES, CAR_FEATURES_ADDED_VALUE, BUYER_3_ACCEPTABLE_RANGE, BUYER_4_ACCEPTABLE_RANGE, TRAIN_BRANDS, DEFAULT_BRANDS, TRAIN_TYPES, DEFAULT_TYPES, VAL_TYPES, VAL_BRANDS, TEST_BRANDS, TEST_TYPES
from agent_prm.utils.general_utils import save_json, load_json

# Set the seed for reproducibility
np.random.seed(42)

"""==============================================================================================================

Making the general car inventory (one for train and one for val + test)

=============================================================================================================="""
# Step 1: Generate 3 copy of the default car inventory for each brand and type
# first copy: B1, B3
# second copy: B4, B6
for i in range(2):
    if i == 0:
        brands_to_use = DEFAULT_BRANDS
        types_to_use = DEFAULT_TYPES
        features_to_sample_from = TRAIN_FEATURES
    else:
        brands_to_use = DEFAULT_BRANDS
        types_to_use = DEFAULT_TYPES
        features_to_sample_from = DEFAULT_FEATURES

    car_inventory_dict = {brand: {car_type: [{
                    "msrp": CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type],
                    "features": []
            }] for car_type in types_to_use} for brand in brands_to_use}
    for brand in brands_to_use:
        for car_type in types_to_use:
            for _ in range(3):
                features = list(np.random.choice(features_to_sample_from, size=np.random.randint(1, 5), replace=False))
                
                msrp_with_features = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type]+ sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in features])
                car_inventory_dict[brand][car_type].append({
                    "msrp": msrp_with_features,
                    "features": features
                })

    save_json(f"src/agent_prm/envs/car_dealer/car_inventory_dict_{i}.json", car_inventory_dict)

# Step 2: For B1, B3, B4, B6
"""
Format:
buyer_idx:
    brand:
        type: 
            budget_1:
            {
                "preferred_brand": brand,
                "preferred_type": type,
                "features": random features,
                "msrp": msrp,
                "budget": budget
            }
            budget_2:
            {
                "preferred_brand": brand,
                "preferred_type": type,
                "features": random features,
                "msrp": msrp,
                "budget": budget
            }
"""
buyer_info_dict = {buyer_idx: {brand: {car_type: {} for car_type in (DEFAULT_TYPES if buyer_idx < 4 else TEST_TYPES)} for brand in (DEFAULT_BRANDS if buyer_idx < 4 else TEST_BRANDS)} for buyer_idx in range(1, 7)}

train_car_inventory_dict = load_json("src/agent_prm/envs/car_dealer/car_inventory_dict_0.json")
val_test_car_inventory_dict = load_json("src/agent_prm/envs/car_dealer/car_inventory_dict_1.json")

def generate_budget(info_dict, features_to_sample_from, buyer_idx, brand, car_type, car_inventory_dict):
    """
    Return:
        the updated info_dict
    """
    for _ in range(2):
        random_features = list(np.random.choice(features_to_sample_from, size=np.random.randint(1, 4), replace=False))
        msrp = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type] + sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in random_features])

        cheapest_car_in_brand = min([car["msrp"] for car in car_inventory_dict[brand][car_type]])
        median_car_in_brand = sorted([car["msrp"] for car in car_inventory_dict[brand][car_type]])[len(car_inventory_dict[brand][car_type]) // 2]
        most_expensive_car_in_brand = max([car["msrp"] for car in car_inventory_dict[brand][car_type]])
   
        if buyer_idx == 1:
            # Use car inventory to generate budget 
            raw_min_budget = int(0.9 * cheapest_car_in_brand)
            raw_max_budget = median_car_in_brand
        elif buyer_idx == 3:
            # Use car inventory to generate budget
            # flexible around budget
            raw_min_budget = int(0.9 * cheapest_car_in_brand) - BUYER_3_ACCEPTABLE_RANGE
            raw_max_budget = median_car_in_brand + BUYER_3_ACCEPTABLE_RANGE
        elif buyer_idx == 4:
            # Use car inventory to generate budget
            # flexible around budget
            raw_min_budget = int(0.9 * median_car_in_brand) - BUYER_4_ACCEPTABLE_RANGE
            raw_max_budget = median_car_in_brand + BUYER_4_ACCEPTABLE_RANGE
        elif buyer_idx == 6:
            # Use car inventory to generate budget
            # Want expensive car
            raw_min_budget = median_car_in_brand
            raw_max_budget = most_expensive_car_in_brand
        else:
            raise ValueError(f"Invalid buyer index: {buyer_idx}")
        
        min_budget = math.ceil(raw_min_budget / 1000) * 1000
        max_budget = math.floor(raw_max_budget / 1000) * 1000  # round down to nearest thousand

        # Step 2: Generate all possible thousand-dollar budgets in the range
        possible_budgets = np.arange(min_budget, max_budget + 1, 1000)

        # Step 3: Randomly select one
        budget = int(np.random.choice(possible_budgets))

        info_dict[buyer_idx][brand][car_type][budget] = {
            "preferred_brand": brand,
            "preferred_type": car_type,
            "features": random_features,
            "msrp": msrp,
            "budget": budget
        }
    return info_dict

for buyer_idx in [1, 3, 4, 6]:
    if buyer_idx in [1, 3]:
        # Train set
        for brand in TRAIN_BRANDS:
            for car_type in TRAIN_TYPES:
                buyer_info_dict = generate_budget(buyer_info_dict, TRAIN_FEATURES, buyer_idx, brand, car_type, car_inventory_dict=train_car_inventory_dict)

        # Val set
        for brand in VAL_BRANDS:
            for car_type in VAL_TYPES:
                buyer_info_dict = generate_budget(buyer_info_dict, DEFAULT_FEATURES, buyer_idx, brand, car_type, car_inventory_dict=val_test_car_inventory_dict)
    else:
        # Test set
        for brand in TEST_BRANDS:
            for car_type in TEST_TYPES:
                buyer_info_dict = generate_budget(buyer_info_dict, DEFAULT_FEATURES, buyer_idx, brand, car_type, car_inventory_dict=val_test_car_inventory_dict)
                

save_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json", buyer_info_dict)

"""==============================================================================================================

Making the one feature data (for B2)

=============================================================================================================="""
def generate_car_inventory_for_features(brands_to_use: List[str], types_to_use: List[str], features_to_sample_from: List[str], features_in_stock: List[str], brand_to_exclude: str, type_to_exclude: str):
    """
    Parameters:
        brands_to_use: List[str]
        types_to_use: List[str]
        features_to_sample_from: List[str] (assuming that this has already removed the 2 features that the user wants)
        feature_in_stock: str
        brand_to_exclude: str
        type_to_exclude: str
    """
    car_inventory_dict = {brand: {car_type: [{
        "msrp": CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type],
        "features": []
    }] for car_type in types_to_use} for brand in brands_to_use}

    # Generate 4-6 car that don't have the same brand and type as the user (but has the same feature and maybe more features)
    cars_with_right_feature_prices = []
    for _ in range(np.random.randint(4, 6)):
        brand_to_sample_from = [brand for brand in brands_to_use if brand != brand_to_exclude]
        type_to_sample_from = [car_type for car_type in types_to_use if car_type != type_to_exclude]

        # Generate a random brand and type that is not the user's brand and type
        random_brand = np.random.choice(brand_to_sample_from)
        random_type = np.random.choice(type_to_sample_from)
        # Generate additional features
        if len(features_in_stock) == 1:
            max_num_features = 4 - len(features_in_stock)
            random_additional_features = list(np.random.choice(features_to_sample_from, size=np.random.randint(0, max_num_features), replace=False))
            random_features = features_in_stock + random_additional_features
        else:
            random_features = features_in_stock

        msrp_with_features = CAR_PRICES_BY_BRAND_AND_TYPE[random_brand][random_type] + sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in random_features])
        car_inventory_dict[random_brand][random_type].append({
            "msrp": msrp_with_features,
            "features": random_features
        })
        cars_with_right_feature_prices.append(msrp_with_features)

    # For all the remaining brand and types, generate enough car until there are at least 4 cars
    for brand in brands_to_use:
        for car_type in types_to_use:
            num_cars_to_generate = 4 - len(car_inventory_dict[brand][car_type])
            for _ in range(num_cars_to_generate):
                random_features = list(np.random.choice(features_to_sample_from, size=np.random.randint(1, 5), replace=False))
                msrp_with_features = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type] + sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in random_features])
                car_inventory_dict[brand][car_type].append({
                    "msrp": msrp_with_features,
                    "features": random_features
                })

    return car_inventory_dict, cars_with_right_feature_prices

def generate_budget_features(info_dict, buyer_idx, brand, car_type, user_features, cars_with_right_feature_prices):
    """
    Return:
        the updated info_dict
    """
    for _ in range(2):
        msrp = CAR_PRICES_BY_BRAND_AND_TYPE[brand][car_type] + sum([CAR_FEATURES_ADDED_VALUE[feature] for feature in user_features])

        cheapest_car_in_brand = min(cars_with_right_feature_prices)
        median_car_in_brand = sorted(cars_with_right_feature_prices)[len(cars_with_right_feature_prices) // 2]
   
        if buyer_idx == 2 or buyer_idx == 5:
            # Must be under budget
            raw_min_budget = int(0.9 * cheapest_car_in_brand)
            raw_max_budget = median_car_in_brand
        else:
            raise ValueError(f"Invalid buyer index: {buyer_idx}")
        
        min_budget = math.ceil(raw_min_budget / 1000) * 1000
        max_budget = math.floor(raw_max_budget / 1000) * 1000  # round down to nearest thousand

        # Step 2: Generate all possible thousand-dollar budgets in the range
        possible_budgets = np.arange(min_budget, max_budget + 1, 1000)

        # Step 3: Randomly select one
        budget = int(np.random.choice(possible_budgets))

        info_dict[buyer_idx][brand][car_type][budget] = {
            "preferred_brand": brand,
            "preferred_type": car_type,
            "features": user_features,
            "msrp": msrp,
            "budget": budget
        }
    return info_dict

# For each brand and type, generate 2 features 
def generate_features_data(info_dict, buyer_idx, data_split, brands_to_use, types_to_use, features_to_sample_from):
    brands_to_gen_inventory_for = TRAIN_BRANDS if data_split == "train" else DEFAULT_BRANDS
    types_to_gen_inventory_for = TRAIN_TYPES if data_split == "train" else DEFAULT_TYPES
    for brand in brands_to_use:
        for car_type in types_to_use:
            num_features_to_sample = np.random.randint(2, 5)
            num_features_in_stock = 1 if buyer_idx == 2 else num_features_to_sample

            # For each user, generate 2 features
            random_features = list(np.random.choice(features_to_sample_from, size=num_features_to_sample, replace=False))
            # Make the first feature the feature in stock
            feature_in_stock = random_features[:num_features_in_stock]

            # Generate a car inventory (that doesn't have the same brand and type as the user)
            features_without_stock = [feature for feature in features_to_sample_from if feature not in random_features]
            car_inventory_dict, cars_with_right_feature_prices = generate_car_inventory_for_features(brands_to_gen_inventory_for, types_to_gen_inventory_for, features_without_stock, feature_in_stock, brand, car_type)

            # For each user, generate 2 budgets (based on the car inventory)
            info_dict = generate_budget_features(info_dict, buyer_idx, brand, car_type, random_features, cars_with_right_feature_prices=cars_with_right_feature_prices)

            # Save the car inventory
            os.makedirs(f"src/agent_prm/envs/car_dealer/buyer_{buyer_idx}_car_inventory", exist_ok=True)
            save_json(f"src/agent_prm/envs/car_dealer/buyer_{buyer_idx}_car_inventory/{brand}_{car_type}_inventory.json", car_inventory_dict)

    return info_dict

# Step 3: Generate the data for B2
# Train set
buyer_info_dict = generate_features_data(buyer_info_dict, 2, "train", TRAIN_BRANDS, TRAIN_TYPES, TRAIN_FEATURES)
# Val set
buyer_info_dict = generate_features_data(buyer_info_dict, 2, "val", VAL_BRANDS, VAL_TYPES, DEFAULT_FEATURES)

save_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json", buyer_info_dict)

# Step 4: Generate the data for B5
# Test set
buyer_info_dict = generate_features_data(buyer_info_dict, 5, "test", TEST_BRANDS, TEST_TYPES, DEFAULT_FEATURES)

save_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json", buyer_info_dict)

