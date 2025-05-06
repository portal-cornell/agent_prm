import re
import ast
from typing import Tuple
from jinja2 import Template

with open("prompts/car_dealer/car_dealer_api_template.j2", "r") as f:
    api_template = Template(f.read())

with open("prompts/car_dealer/car_dealer_template.j2", "r") as f:
    response_template = Template(f.read())

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

def format_reason_action_car_dealer_critic(query: dict) -> str:
    """
    Used by the critic to format the query so that the reason action can be rendered in the prompt.
    
    Parameters:
        query: a dictionary containing the query and the reason action
            size: batch_size

    Returns:
        a dictionary containing the query and the reason action (ready to be rendered in the prompt)
    """
    mode = "api_call" if "api_name" in query["action"].keys() else "generate_response"

    action = query["action"]

    if mode == "generate_response":
        """
        This part belongs to the reason action when it's passed in as a query
        {
            "response": response,
            "car_idx": car_idx,
            "proposed_car": {
                "brand": proposed_car_brand,
                "type": proposed_car_type,
                "features": proposed_car_features,
                "msrp": proposed_car_msrp
            }
        }

        We need to get "proposed_car_brand", "proposed_car_type", "proposed_car_features", and "proposed_car_msrp" from the query
        """
        query["response"] = action["response"]
        query["car_idx"] = action["car_idx"]
        query["proposed_car_brand"] = action["proposed_car"]["brand"] if action["proposed_car"]["brand"] != "" else "None"
        query["proposed_car_type"] = action["proposed_car"]["type"] if action["proposed_car"]["type"] != "" else "None"
        query["proposed_car_features"] = action["proposed_car"]["features"]
        query["proposed_car_msrp"] = action["proposed_car"]["msrp"]
    elif mode == "api_call":
        """
        This part belongs to the reason action when it's passed in as a query
        """
        query["api_name"] = action["api_name"] if action["api_name"] != "" else "no_op"
        query["api_brand"] = action["api_brand"] if action["api_brand"] != "" else "None"
        query["api_type"] = action["api_type"] if action["api_type"] != "" else "None"
        query["api_features"] = action["api_features"] if action["api_features"] != "" else []

    return query

def format_reason_action_car_dealer_online_dpo(text: str) -> str:
    """
    Used in online DPO to format the reason and action in the same format as the prompt.
    """
    mode = "api_call" if "API NAME" in text or "API BRAND" in text or "API TYPE" in text or "API FEATURES" in text else "generate_response"

    # Add other check for the mode
    parsed_api_reason, parsed_api_action = parse_reason_and_action_car_dealer_api_call(text)
    parsed_response_reason, parsed_response_action = parse_reason_and_action_car_dealer(text)

    if mode == "api_call":
        # Warning: if it's an api call BUT its parsed reason is empty but the parsed response reason is not empty (it implies that this response is actually a generate_response)
        if parsed_api_reason == "" and parsed_response_reason != "":
            # print(f"Warning: this response is actually a generate_response but it's parsed as an api call")
            mode = "generate_response"
    elif mode == "generate_response":
        # Warning: if it's a generate_response BUT its parsed reason is not empty but the parsed api reason is empty (it implies that this response is actually an api call)
        if parsed_response_reason == "" and parsed_api_reason != "":
            # print(f"Warning: this response is actually an api call but it's parsed as a generate_response")
            mode = "api_call"

    if mode == "api_call":
        reason = parsed_api_reason
        action = parsed_api_action
    elif mode == "generate_response":
        reason = parsed_response_reason
        action = parsed_response_action

    if reason == "":
        output_response = f"<|eot_id|>" # A trick to penalize ill-formed responses that cannot be parsed
    else:
        if mode == "api_call":
            output_response = api_template.render(system=False, mode="output", 
                                                    reason=reason,
                                                    api_name=action["api_name"],
                                                    api_brand=action["api_brand"],
                                                    api_type=action["api_type"],
                                                    api_features=action["api_features"]
                                                    ).strip()
        elif mode == "generate_response":
            output_response = response_template.render(system=False, mode="output", 
                                                        reason=reason,
                                                        response=action["response"],
                                                        car_idx=action["car_idx"],
                                                        proposed_car_brand=action["proposed_car"]["brand"],
                                                        proposed_car_type=action["proposed_car"]["type"],
                                                        proposed_car_features=action["proposed_car"]["features"],
                                                        proposed_car_msrp=action["proposed_car"]["msrp"]
                                                        ).strip()

        output_response = output_response + "<|eot_id|>"

    # print("================================================")
    # print(text)
    # print(output_response)
    # print("================================================")

    return output_response