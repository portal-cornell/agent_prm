import re
import ast
from typing import Tuple

def parse_reason_and_action_car_dealer_api_call(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for car_dealer environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action. 
        reason: The reason for the action.
        action: A dictionary containing the api call.
        {
            "api_name": "string: the name of the API call",
            "api_brand": "string: the brand of the car",
            "api_type": "string: the type of the car"
        }
    """
    # Process the text so that the key words are guaranteed to be captialized
    lower_text = text.lower()

    for keyword in ["reason:", "api name:", "api brand:", "api type:", "api features:"]:
        # Find the index for the first occurence of "reason:"
        idx = lower_text.find(keyword)
        if idx != -1:
            # Replace the key words in the original text with the capitalized key words
            text = text[:idx] + keyword.upper() + text[idx+len(keyword):]

    # API call
    pattern = r"REASON\s*:?\s*([\s\S]*?)\s*API NAME\s*:?\s*(.*?)\s*API BRAND\s*:?\s*(.*?)\s*API TYPE\s*:?\s*(.*?)\s*API FEATURES\s*:?\s*(.*)"

    match = re.search(pattern, text)

    if match:
        reason = match.group(1).strip()  # Remove extra spaces/newlines
        api_name = match.group(2).strip()
        api_brand = match.group(3).strip()
        api_type = match.group(4).strip()
        api_features = match.group(5).strip()

        try:
            api_features = ast.literal_eval(api_features)
        except Exception as e:
            api_features = []

        api_brand = "" if api_brand == "None" else api_brand
        api_type = "" if api_type == "None" else api_type

        if api_name not in ["search_car_by_brand_type", "search_car_by_brand", "search_car_by_type", "search_car_that_have_features", "no_op"]:
            api_name = "no_op"
            api_brand = ""
            api_type = ""
            api_features = []
    else:
        reason = ""
        api_name = "no_op"
        api_brand = ""
        api_type = ""
        api_features = []

    action = {
        "api_name": api_name,
        "api_brand": api_brand,
        "api_type": api_type,
        "api_features": api_features
    }

    return reason, action

def parse_reason_and_action_car_dealer(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for car_dealer environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action. 
        reason: The reason for the action.
        action: A dictionary containing the response and the car index.
        {
            "response": "string: the response to the buyer",
            "car_idx": "integer: the index of the car you are proposing to the buyer (0 if you haven't looked up any car yet)"
        }
    """
    # Process the text so that the key words are guaranteed to be captialized
    lower_text = text.lower()
    for keyword in ["reason:", "response:", "car index:", "proposed car brand:", "proposed car type:", "proposed car features:", "proposed car msrp:"]:
        # Find the index for the first occurence of "reason:"
        idx = lower_text.find(keyword)
        if idx != -1:
            # Replace the key words in the original text with the capitalized key words
            text = text[:idx] + keyword.upper() + text[idx+len(keyword):]

    # API call
    pattern = r"REASON\s*:?\s*([\s\S]*?)\s*RESPONSE\s*:?\s*(.*?)\s*CAR INDEX\s*:?\s*(.*?)\s*PROPOSED CAR BRAND\s*:?\s*(.*?)\s*PROPOSED CAR TYPE\s*:?\s*(.*?)\s*PROPOSED CAR FEATURES\s*:?\s*(.*?)\s*PROPOSED CAR MSRP\s*:?\s*(.*)"
    match = re.search(pattern, text)

    if match:
        reason = match.group(1).strip()  # Remove extra spaces/newlines
        response = match.group(2).strip()
        try:
            car_idx = int(match.group(3).strip())
        except Exception as e:
            car_idx = 0
        proposed_car_brand = match.group(4).strip()
        proposed_car_type = match.group(5).strip()

        try:
            proposed_car_features = ast.literal_eval(match.group(6).strip())
        except Exception as e:
            proposed_car_features = []

        try:
            proposed_car_msrp = int(match.group(7).strip())
        except Exception as e:
            proposed_car_msrp = 0
    else:
        reason = ""
        response = ""
        car_idx = 0
        proposed_car_brand = ""
        proposed_car_type = ""
        proposed_car_features = []
        proposed_car_msrp = 0

    action = {
        "response": response,
        "car_idx": car_idx,
        "proposed_car": {
            "brand": proposed_car_brand,
            "type": proposed_car_type,
            "features": proposed_car_features,
            "msrp": proposed_car_msrp
        }
    }

    return reason, action
