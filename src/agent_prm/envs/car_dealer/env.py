"""
Adapted from https://github.com/abdulhaim/LMRL-Gym
"""
import random
import time
from typing import Dict, List, Optional, Tuple
from agent_prm.envs.car_dealer.simulator import SGLangServerCarDealerSimulator
from agent_prm.envs.car_dealer.data import extract_final_decision_from_buyer_reply, compute_reward, check_has_discount, determine_car_inventory

def are_same_cars(car1: Dict, car2: Dict) -> bool:
    """
    Check if two cars are the same.
    """
    if car1 == {} and car2 == {}:
        return True
    elif car1 == {} or car2 == {}:
        return False

    return car1["brand"] == car2["brand"] and car1["type"] == car2["type"] and sorted(car1["features"]) == sorted(car2["features"])

class CarDealerEnvironment():
    """
    A state less environment for the 20 questions game.
        (The environment will not track the conversation so far)
    """
    def __init__(
        self, 
        buyer: SGLangServerCarDealerSimulator,
        reward_mode: str="fancy",
        max_conversation_length: int=10,
    ):
        self.buyer = buyer
        self.max_conversation_length = max_conversation_length
        self.reward_mode = reward_mode

        self.random = random.Random(None)
               

    def step(self, buyer_info: Dict, history: List[Dict], action: str, seller_proposed_car: dict, proposed_car_copied_in_response: dict, num_negotiation: int, prev_proposed_car: dict, num_car_proposed: int, car_inventories: dict):
        """
        Parameters:
            buyer_info (Dict): The information about the buyer.
            {
                "buyer_strategy": str,
                "preferred_brand": str,
                "preferred_type": str,
                "preferred_features": List[str],
                "budget": int,
                "msrp": int
            }
            history (List[Dict]): The history of the conversation so far. A list of dictionaries. 
            [
                {
                    "role": str ("buyer" or "seller"),
                    "content": str,
                }
            ]
            action (str): The text to say to the buyer.

        Returns:
            history (List[Dict]): The updated history of the conversation so far (with the new text)
            buyer_reason (str): The reason the buyer said the text.
            buyer_text (str): The text the buyer said.
            reward (float): The reward for the action.
            done (bool): Whether the conversation is done.
        """
        # Update num_negotiations and num_car_proposed
        has_discount, _ = check_has_discount(action, seller_proposed_car)
        same_car = are_same_cars(prev_proposed_car, seller_proposed_car)
        print(f"has_discount: {has_discount}, same_car: {same_car}")

        if not same_car:
            num_car_proposed += 1
            num_negotiation += 1 # Proposing a new car also counts as a negotiation
        else:
            if has_discount:
                num_negotiation += 1

        start_time = time.time()
        buyer_reason, buyer_response, buyer_decision = self.buyer.generate_response(buyer_info, history, action, seller_proposed_car, num_negotiation, num_car_proposed)
        end_time = time.time()
        print(f"Time taken to generate answer: {end_time - start_time} seconds")
        print("-"*100)
        print(f"num_negotiation: {num_negotiation}, num_car_proposed: {num_car_proposed}")
        print(f"Buyer reason:\n{buyer_reason}")
        print(f"Buyer text:\n{buyer_response}")
        print(f"Buyer decision:\n{buyer_decision}")
        print("-"*100)

        history.append({
            "role": "seller",
            "content": action,
        })

        history.append({
            "role": "buyer",
            "content": buyer_response,
        })

        final_decision = extract_final_decision_from_buyer_reply(buyer_decision)
        # Compute the reward
        if final_decision is not None:
            # Determine the car inventory to use
            car_inventory_dict = determine_car_inventory(buyer_info, car_inventories)
            reward, success, failure_reason = compute_reward(buyer_info, final_decision, seller_proposed_car, proposed_car_copied_in_response, num_negotiation, num_car_proposed, car_inventory_dict, self.reward_mode)
            done = True
        else:
            reward = 0.0 # Punish the seller for not expediting the conversation
            success = False
            done = False
            failure_reason = "Have not made a decision yet."

        if len(history)//2 >= self.max_conversation_length:  # Because the history is doubled (buyer and seller)
            done = True
        
        return history, buyer_reason, buyer_response, buyer_decision, reward, success, failure_reason, done, num_negotiation, seller_proposed_car, num_car_proposed
    
    def reset(self):
        """
        Returns:
            history (List[Dict]): The history of the conversation so far (in the beginning, it's empty). A list of dictionaries, of the form:
            {
                "role": str,
                "content": str,
            }
        """
        return []


class BatchedCarDealerEnvironment():
    """
    A batched environment for the car dealer game.
    """
    def __init__(
        self, 
        buyer: SGLangServerCarDealerSimulator,
        reward_mode: str="fancy",
        max_conversation_length: int=10,
    ):
        self.buyer = buyer
        self.max_conversation_length = max_conversation_length
        self.reward_mode = reward_mode

        self.random = random.Random(None)
               

    def step(self, buyer_infos: List[Dict], histories: List[List[Dict]], actions: List[str], seller_proposed_cars: List[Dict], proposed_cars_copied_in_responses: List[Dict], num_negotiations: List[int], prev_proposed_cars: List[Dict], num_car_proposed: List[int], car_inventories: dict, prev_dones: List[bool]):
        """
        Parameters:
            buyer_info (Dict): The information about the buyer.
            {
                "buyer_strategy": str,
                "preferred_brand": str,
                "preferred_type": str,
                "preferred_features": List[str],
                "budget": int,
                "msrp": int
            }
            history (List[Dict]): The history of the conversation so far. A list of dictionaries. 
            [
                {
                    "role": str ("buyer" or "seller"),
                    "content": str,
                }
            ]
            action (str): The text to say to the buyer.

        Returns:
            history (List[Dict]): The updated history of the conversation so far (with the new text)
            buyer_reason (str): The reason the buyer said the text.
            buyer_text (str): The text the buyer said.
            reward (float): The reward for the action.
            done (bool): Whether the conversation is done.
            num_negotiations (int): The number of negotiations that the buyer has had with the seller.
            num_car_proposed (int): The number of cars that the seller has proposed to the buyer.
        """
        # Update num_negotiations and num_car_proposed
        for i in range(len(histories)): # For each game
            if not prev_dones[i]:
                has_discount, _ = check_has_discount(actions[i], seller_proposed_cars[i])
                same_car = are_same_cars(prev_proposed_cars[i], seller_proposed_cars[i])

                if not same_car:
                    num_car_proposed[i] += 1
                    num_negotiations[i] += 1 # Proposing a new car also counts as a negotiation
                else:
                    if has_discount:
                        num_negotiations[i] += 1

        # Get batched answer
        start_time = time.time()
        buyer_reasons, buyer_responses, buyer_decisions = self.buyer.generate_response_batch(buyer_infos, histories, actions, seller_proposed_cars, num_negotiations, num_car_proposed)
        end_time = time.time()
        print(f"[ENV] Time taken to generate answer: {end_time - start_time} seconds")

        # Update histories and compute rewards
        rewards = []
        successes = []
        failure_reasons = []
        dones = []
        for i in range(len(histories)):
            if not prev_dones[i]:
                histories[i].append({
                    "role": "seller",
                    "content": actions[i],
                })

                histories[i].append({
                    "role": "buyer",
                    "content": buyer_responses[i],
                })

                final_decision = extract_final_decision_from_buyer_reply(buyer_decisions[i])
                # Compute the reward
                if final_decision is not None:
                    # Determine the car inventory to use
                    car_inventory_dict = determine_car_inventory(buyer_infos[i], car_inventories)
                    reward, success, failure_reason = compute_reward(buyer_infos[i], final_decision, seller_proposed_cars[i], proposed_cars_copied_in_responses[i], num_negotiations[i], num_car_proposed[i], car_inventory_dict, self.reward_mode)
                    done = True
                else:
                    reward = 0.0 # Punish the seller for not expediting the conversation
                    success = False
                    done = False
                    failure_reason = "Have not made a decision yet."

                if len(histories[i])//2 >= self.max_conversation_length:  # Because the history is doubled (buyer and seller)
                    done = True

                rewards.append(reward)
                successes.append(success)
                failure_reasons.append(failure_reason)
                dones.append(done)
            else:
                rewards.append(0.0)
                successes.append(False)
                failure_reasons.append("Game is done.")
                dones.append(True)
                
        return histories, buyer_reasons, buyer_responses, buyer_decisions, rewards, successes, failure_reasons, dones, num_negotiations, seller_proposed_cars, num_car_proposed
    
    def reset(self, buyer_infos: List[Dict]):
        """
        Returns:
            history (List[Dict]): The history of the conversation so far (in the beginning, it's empty). A list of dictionaries, of the form:
            {
                "role": str,
                "content": str,
            }
        """
        return [[] for _ in range(len(buyer_infos))]

def setup_car_dealer_env(host: str = 'localhost', port: int = 40042) -> CarDealerEnvironment:
    env = CarDealerEnvironment(
        buyer=SGLangServerCarDealerSimulator(
            model_id="Qwen/Qwen2.5-14B-Instruct",
            server_url=f"http://{host}:{port}",
            prompt_template_file="prompts/car_dealer/car_dealer_simulator_template_with-reasoning.j2",
            verbose=0
        )
    #     buyer=SGLangServerCarDealerSimulator(
    #         model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    #         server_url=f"http://{host}:{port}",
    #         prompt_template_file="prompts/car_dealer/car_dealer_simulator_template_with-reasoning.j2",
    #         verbose=0
    #     )
    )
    return env

def setup_batched_car_dealer_env(host: str = 'localhost', port: int = 40042) -> BatchedCarDealerEnvironment:
    env = BatchedCarDealerEnvironment(
        buyer=SGLangServerCarDealerSimulator(
            model_id="Qwen/Qwen2.5-14B-Instruct",
            server_url=f"http://{host}:{port}",
            prompt_template_file="prompts/car_dealer/car_dealer_simulator_template_with-reasoning.j2",
            verbose=0
        )
        # buyer=SGLangServerCarDealerSimulator(
        #     model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #     server_url=f"http://{host}:{port}",
        #     prompt_template_file="prompts/car_dealer/car_dealer_simulator_template_with-reasoning.j2",
        #     verbose=0
        # )
    )
    return env
