import copy
from typing import List

from agent_prm.utils.general_utils import load_json

car_inventory_dict = load_json("src/agent_prm/envs/car_dealer/car_inventory_dict.json")

def use_api(arguments: dict):
    """
    Return a list of cars that match the given arguments.
    [

    ]
    """
    api_brand = arguments["api_brand"].strip().lower().capitalize()
    api_type = arguments["api_type"].strip().lower()

    if arguments["api_name"] == "search_car_by_brand_type":
        return search_car_by_brand_type(api_brand, api_type)
    elif arguments["api_name"] == "search_car_by_brand":
        return search_car_by_brand(api_brand)
    elif arguments["api_name"] == "search_car_by_type":
        return search_car_by_type(api_type)
    elif arguments["api_name"] == "no_op":
        return []
    else:
        print(f"Invalid API name: {arguments['api_name']}")
        return []

def search_car_by_brand_type(brand: str, car_type: str):
    if car_type not in car_inventory_dict[brand]:
        return []

    car_list = copy.deepcopy(car_inventory_dict[brand][car_type])

    # Add the car's brand and car type to the car list
    for car in car_list:
        car["brand"] = brand
        car["type"] = car_type

    return car_list

def search_car_by_brand_type_features(brand: str, car_type: str, features: List[str]):
    """
    Legacy API.

    Return a list of cars that match the given brand, car type, and features.
    """
    car_by_brand_type = search_car_by_brand_type(brand, car_type)

    sorted_features = sorted(features)
    matching_cars = []
    for car in car_by_brand_type:
        car_features = sorted(car["features"])
        if car_features == sorted_features:
            matching_cars.append(car)
    return matching_cars

def search_car_by_brand(brand: str):
    car_list = []
    for car_type in car_inventory_dict[brand]:
        for car in car_inventory_dict[brand][car_type]:
            car_info = copy.deepcopy(car)
            # Add the car's brand and car type to the car info
            car_info["brand"] = brand
            car_info["type"] = car_type
            car_list.append(car_info)
    return car_list

def search_car_by_type(car_type: str):
    car_list = []
    for brand in car_inventory_dict:
        if car_type in car_inventory_dict[brand]:
            for car in car_inventory_dict[brand][car_type]:
                car_info = copy.deepcopy(car)
                car_info["brand"] = brand
                car_info["type"] = car_type
                car_list.append(car_info)
    return car_list
    