import re
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

    for keyword in ["reason:", "api name:", "api brand:", "api type:"]:
        # Find the index for the first occurence of "reason:"
        idx = lower_text.find(keyword)
        if idx != -1:
            # Replace the key words in the original text with the capitalized key words
            text = text[:idx] + keyword.upper() + text[idx+len(keyword):]

    # API call
    pattern = r"REASON:\s*([\s\S]*?)\s*API NAME:\s*(.*?)\s*API BRAND:\s*(.*?)\s*API TYPE:\s*(.*)"
    match = re.search(pattern, text)

    if match:
        reason = match.group(1).strip()  # Remove extra spaces/newlines
        api_name = match.group(2).strip()
        api_brand = match.group(3).strip()
        api_type = match.group(4).strip()

        api_brand = "" if api_brand == "None" else api_brand
        api_type = "" if api_type == "None" else api_type

        if api_name not in ["search_car_by_brand_type", "search_car_by_brand", "search_car_by_type", "no_op"]:
            api_name = "no_op"
            api_brand = ""
            api_type = ""
    else:
        reason = ""
        api_name = "no_op"
        api_brand = ""
        api_type = ""

    action = {
        "api_name": api_name,
        "api_brand": api_brand,
        "api_type": api_type
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
    for keyword in ["reason:", "response:", "car index:"]:
        # Find the index for the first occurence of "reason:"
        idx = lower_text.find(keyword)
        if idx != -1:
            # Replace the key words in the original text with the capitalized key words
            text = text[:idx] + keyword.upper() + text[idx+len(keyword):]

    # API call
    pattern = r"REASON:\s*([\s\S]*?)\s*RESPONSE:\s*(.*?)\s*CAR INDEX:\s*(.*)"
    match = re.search(pattern, text)

    if match:
        try:
            reason = match.group(1).strip()  # Remove extra spaces/newlines
            response = match.group(2).strip()
            car_idx = int(match.group(3).strip())
        except Exception as e:
            reason = "reason"
            response = "response"
            car_idx = 0
    else:
        reason = ""
        response = ""
        car_idx = 0

    action = {
        "response": response,
        "car_idx": car_idx
    }

    return reason, action
