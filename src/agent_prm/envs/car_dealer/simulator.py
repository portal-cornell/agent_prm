import re
import requests
import json
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Dict, List
from jinja2 import Template
from transformers import AutoTokenizer

from agent_prm.envs.car_dealer.data import format_chat_history, format_car_suggestion, get_mode_and_curr_info_and_buyer_strategy

def parse_reason_and_action_car_dealer_simulator(text: str) -> Tuple[str, str]:
    """
    Parses the reason and action given prediction from model for ORCALE environment 

    Args:
        text: The text containing the reason and action.

    Returns:
        A tuple with the parsed reason and action. 
    """
    pattern = r"REASON:\s*([\s\S]*?)\s*ANSWER:\s*([\s\S]*?)\s*DECISION:\s*([\s\S]*?)$"
    match = re.search(pattern, text)

    if match:
        reason = match.group(1).strip()  # Remove extra spaces/newlines
        answer = match.group(2).strip()
        decision = match.group(3).strip()
    else:
        reason = "None"
        answer = "Something went wrong on the buyer's side, so they are not able to respond to you. Please wait."
        decision = "None"

    return reason, answer, decision


class SGLangServerCarDealerSimulator(object):
    """
    Initialize the CarDealerSimulator agent.
    """
    def __init__(self, 
                 model_id: str, 
                 server_url: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = parse_reason_and_action_car_dealer_simulator, 
                 max_tokens: int = 1024,
                 batch_limit: Optional[int] = None) -> None:
        self.model_id = model_id
        self.server_url = server_url.rstrip('/') + '/generate'
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

        self.max_tokens = max_tokens
        self.batch_limit = batch_limit
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # TODO: hardcoded for now
        self.buyer_curr_info_templates = {}
        for i in range(1, 7):
            with open(f"prompts/car_dealer/car_dealer_simulator_buyer_info_{i}.j2", "r") as file:
                self.buyer_curr_info_templates[i] = Template(file.read())

    def generate_response(self,
                          buyer_info: Dict,
                          history: List[Dict],
                          seller_response: str,
                          seller_proposed_car: Dict = {},
                          num_negotiation: int = 0,
                          num_car_proposed: int = 0) -> Tuple[str, str]:
        """
        Generate a response based on the current chat history and buyer information
        """
        buyer_reasons, buyer_responses, buyer_decisions = self.generate_response_batch([buyer_info], [history], [seller_response], [seller_proposed_car], [num_negotiation], [num_car_proposed])
        return buyer_reasons[0], buyer_responses[0], buyer_decisions[0]

    def generate_response_batch(self,
                          buyer_infos: List[Dict],
                          histories: List[List[Dict]],
                          seller_responses: List[str],
                          seller_proposed_cars: List[Dict] = [],
                          num_negotiations: List[int] = [],
                          num_car_proposed: List[int] = []) -> Tuple[str, str]:
        """
        Generate a response based on the current chat history and buyer information

        Parameters:
            histories: The chat history of the conversation so far.
            buyer_info: The information about the buyer.
            [
                {
                    "buyer_strategy": str,
                    "preferred_brand": str,
                    "preferred_type": str,
                    "preferred_features": List[str],
                    "budget": int,
                    "msrp": int
                }
            ]
            curr_car_of_interest: The current car of interest.
            [
                {
                    "brand": str,
                    "type": str,
                    "features": List[str],
                    "msrp": int
                }
            ]

        Returns:
            A tuple with the parsed reason and action. 
        """
        mode_and_curr_infos_and_buyer_strategies = [
            get_mode_and_curr_info_and_buyer_strategy(
                self.buyer_curr_info_templates[buyer_info["id"]], 
                buyer_info, 
                seller_proposed_car, 
                seller_response, 
                history,
                num_negotiation,
                num_car_proposed
            ) # Tuple[str, str]
            for buyer_info, history, seller_response, seller_proposed_car, num_negotiation, num_car_proposed in zip(buyer_infos, histories, seller_responses, seller_proposed_cars, num_negotiations, num_car_proposed)
        ]

        modes = [x[0] for x in mode_and_curr_infos_and_buyer_strategies]
        curr_infos = [x[1] for x in mode_and_curr_infos_and_buyer_strategies]
        buyer_strategies = [x[2] for x in mode_and_curr_infos_and_buyer_strategies]

        input_datas = [
            {
                "mode": "input",
                "observation_action_history": format_chat_history(history, mode),
                "seller_response": seller_response,
                "dealer_proposed_car": format_car_suggestion(seller_proposed_car),
                "curr_info": curr_info,
                "buyer_strategy": buyer_strategy,
                **buyer_info
            }
            for mode, curr_info, buyer_strategy, history, buyer_info, seller_response, seller_proposed_car in zip(modes, curr_infos, buyer_strategies, histories, buyer_infos, seller_responses, seller_proposed_cars)
        ]

        messages = [
            [
                {"role": "user", "content": self.prompt_template.render(**input_data).strip()}
            ]
            for input_data in input_datas
        ]

        # for i in range(len(messages)):
        #     print(json.dumps(input_datas[i], indent=4))
        #     print(messages[i][0]["content"])
        #     input("message")
        
        batch_limit = self.batch_limit if self.batch_limit is not None else len(messages)
        generated_texts = []

        if self.verbose==1:  
            iterator = tqdm(range(0, len(messages), batch_limit), desc="Querying sglang agent")
        else:
            iterator = range(0, len(messages), batch_limit)
        
        for i in iterator:
            messages_batch = messages[i:i+batch_limit]
            prompts_batch = self.tokenizer.apply_chat_template(messages_batch, tokenize=False, add_generation_prompt=True)
            data_batch = {"model": self.model_id, 
                          "text": prompts_batch,
                          "sampling_params": {
                              "temperature": 0.3,
                              "max_new_tokens": self.max_tokens,
                          },
                          }
            
            responses_batch = requests.post(self.server_url, 
                                            json=data_batch).json()
            generated_texts_batch = [x["text"] for x in responses_batch]
            generated_texts.extend(generated_texts_batch)

        # print(f"generated_texts:\n{generated_texts}")
        
        buyer_reasons, buyer_responses, buyer_decisions = [], [], []
        for i in range(len(generated_texts)):
            reason, answer, decision = self.parse_reason_action_fn(generated_texts[i])
            buyer_reasons.append(reason)
            buyer_responses.append(answer)
            buyer_decisions.append(decision)

        return buyer_reasons, buyer_responses, buyer_decisions
