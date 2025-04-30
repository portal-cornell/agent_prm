"""
Correct all the states and actions of a failed rollout

for each failed rollout,
    - generate a summary
    - generate a correction for each state and action (given state and the summary)

Usage:
    python scripts/dataproc/expert_relabel/correct_rollouts_twenty_questions.py -d train -i 1 -e

    -i iteration, the iteration for the pi that you are training for (e.g., pi3 would be iteration=2)
"""
import argparse
import os
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from jinja2 import Template
from omegaconf import DictConfig, OmegaConf

from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json
from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT, get_default_word_list

ALL_OBJ_LIST = [wv[0] for wv in get_default_word_list("all")]
MAX_QUERY_ATTEMPTS = 3

def relabel_one_rollout(rollout_path: str, expert_correction_prompt: Template, expert_summary_prompt: Template, obj: str, category: str):
    """
    Effect:
        - Generate a summary of the rollout
        - Generate a correction for each state and action (given state and the summary)
    """
    rollout = load_json(rollout_path)
    total_cost = 0

    # Generate a summary
    if "summary" not in rollout[0]:
        summary, cost = gen_summary_from_rollout(expert_summary_prompt, rollout, obj, category)
        rollout[0]["summary"] = summary
        save_json(rollout_path, rollout)
        total_cost += cost
    else:
        summary = rollout[0]["summary"]
        total_cost += 0

    print(f"=========== Summary ===========")
    print(summary)
    print("="*100)

    # Iterate through the rollout and generate a correction for each state and action
    for t in range(len(rollout)):
        if "expert_alternatives" not in rollout[t]:
            rollout[t]["expert_alternatives"] = []  # Edit the rollout file to save the alt actions

            alt_reason_actions_list, cost = gen_alt_actions(expert_correction_prompt, obj, category, summary, rollout, t, num_alt_actions_to_gen=1)

            print(f"t={t}: Alt reason actions (cost: ${cost:.2f}):\n{json.dumps(alt_reason_actions_list, indent=4)}")

            rollout[t]["expert_alternatives"].extend(alt_reason_actions_list)

            save_json(rollout_path, rollout)
            total_cost += cost
        else:
            print(f"t={t}: Skip because already has expert alternatives")

        print(rollout_path) # So it's easier to debug and find the rollout

    print(f"Total cost = ${total_cost} for {rollout_path}")

    return total_cost

def format_chat_history_and_goal(rollout: List[Dict], t: int, secret_word:str, category: str="") -> Tuple[str, str]:
    """
    Return
        - chat_history: str (until t)
        - goal: str (optional, if file_name is provided)
    """
    answer_str = f"The secret word is '{secret_word}'. It's in the general category '{category}'."

    history_str = ""
    for i in range(t):
        history_str += f"Question #{i+1}: {rollout[i]['action']}\nAnswer #{i+1}: {rollout[i]['answer']}\n"

    if history_str == "":
        history_str = "No chat history yet. Just start with the question."

    return history_str, answer_str

def gen_alt_actions(
        expert_correction_prompt: Template,
        # Used to verify the feasibility of the reasoning
        secret_word: str,
        category: str,
        # Used to generate alternative action
        summary: str, 
        rollout: List[Dict], 
        t: int, 
        num_alt_actions_to_gen: int) -> List[Dict]:
    """
    Generate the alternative actions for the given timestep
    """
    chat_history, _ = format_chat_history_and_goal(rollout, t, secret_word, category)
    chat_history = chat_history.strip()

    system_prompt = expert_correction_prompt.render(system=True, all_obj_list=ALL_OBJ_LIST, summary=summary, num_responses=num_alt_actions_to_gen).strip()
    input_prompt = expert_correction_prompt.render(system=False, mode="input", observation_action_history=chat_history, num_responses=num_alt_actions_to_gen).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    terminate = False
    query_attempts = 0
    query_cost = 0

    while not terminate and query_attempts < MAX_QUERY_ATTEMPTS:
        response, cost = generate_from_openai_completion(
            messages=messages, model="gpt-4o", temperature=0.7
        )

        reason_action_list = parse_json(response)
        try:
            assert reason_action_list is not None, f"Failed to parse response: {response}"
            for reason_action in reason_action_list:
                assert "teacher_reason" in reason_action and "player_reason" in reason_action and "question" in reason_action, f"Invalid response: {reason_action}. Must contain 'teacher_reason', 'player_reason' and 'question'"
            terminate = True
        except Exception as e:
            print(f"Error parsing response: {response}")
            raise e
        
        query_attempts += 1
        query_cost += cost

    if not terminate:
        elogger.log(f"Failed to get a valid response after {MAX_QUERY_ATTEMPTS} attempts")
        raise Exception(f"Failed to get a valid response after {MAX_QUERY_ATTEMPTS} attempts")
    
    # Post-process to rename "question" to "action" and "player_reason" to "reason"
    for reason_action in reason_action_list:
        reason_action["action"] = reason_action.pop("question")
        reason_action["reason"] = reason_action.pop("player_reason")

    # Set a flag that the reasoning has potentially secret information
    for reason_action in reason_action_list:
        reason_action["feasibility"] = check_reasoning_feasibility(reason_action["reason"], secret_word, category)

        if reason_action["feasibility"] == "low":
            print(f"=======================")
            print(chat_history)
            print(f"Teacher Reasoning:\n{reason_action['teacher_reason']}")
            print(f"Question:\n{reason_action['action']}")
            print(f"Player Reasoning:\n{reason_action['reason']}")
            print(f"Feasibility: {reason_action['feasibility']}")
            print(f"=======================")
            elogger.log(f"Please verify the reasoning of the following rollout: {reason_action}")
            input("Press Enter to continue...")

    return reason_action_list, query_cost

def check_reasoning_feasibility(reason: str, secret_word: str, category: str) -> Tuple[str, str]:
    """
    Check if the reasoning is infeasible

    Return:
        'high': The reasoning is feasible (didn't have any of the condition that makes it 'medium' or 'low' level of feasibility)
        'medium': The reasoning is potentially feasible, but requires manual verification
        'low': The reasoning is infeasible
    """
    reason_lower = reason.lower()
    secret_word_lower = secret_word.lower()
    category_lower = category.lower()

    if (secret_word_lower in reason_lower) or (category_lower in reason_lower):
        return 'medium'
    elif ("summary" in reason_lower) or (f"the secret word is {secret_word_lower}" in reason_lower) or (f"the secret word is '{secret_word_lower}'" in reason_lower) or (f'the secret word is "{secret_word_lower}"' in reason_lower):
        return 'low'
    else:
        return 'high'
    

def gen_summary_from_rollout(summary_prompt: Template, rollout: List[Dict], secret_word: str, category: str) -> Tuple[str, float]:
    """
    Generate a summary from the rollout
    """
    chat_history, answer_str = format_chat_history_and_goal(rollout, len(rollout), secret_word, category)

    system_prompt = summary_prompt.render(system=True, all_obj_list=ALL_OBJ_LIST).strip()
    input_prompt = summary_prompt.render(system=False, mode="input", observation_action_history=chat_history, goal=answer_str).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(messages=messages, model="gpt-4o", temperature=0.3)

    return response, cost

def get_failed_rollouts(config: DictConfig, data_types: List[str], iteration: int):
    """
    Returns:
        - a list of failed rollout path
    """
    rollout_dir = config.leap[f"iter{iteration - 1}"]["rollout_dir"]
    failed_rollout_paths = []

    # Get the failed rollouts (and the object and the category)
    for data_type in data_types:
        for data_type in data_types:
            if data_type == "train":
                object_dict_to_use = TRAIN_OBJECT_DICT
            elif data_type == "val":
                object_dict_to_use = VALIDATION_OBJECT_DICT

        objects_to_eval_on = [(obj, category, data_type, rollout_idx) for category in object_dict_to_use.keys() for obj in object_dict_to_use[category] for rollout_idx in range(config.leap.rollout_per_task_range_min, config.leap.rollout_per_task_range_max)]

        for obj, category, _, rollout_idx in tqdm(objects_to_eval_on, desc="Processing objects"):
            rollout_path = os.path.join(rollout_dir, data_type, f"{obj}_{rollout_idx}.json")
            if is_failed_rollout(rollout_path):
                failed_rollout_paths.append((rollout_path, obj, category))

    return failed_rollout_paths


def is_failed_rollout(rollout_path: str):
    """
    Effect:
        - Check if the rollout is failed
    """
    rollout = load_json(rollout_path)
    return rollout[-1]["reward"] != 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/create_sft_training_data/twenty_questions.yaml")
    parser.add_argument("-d", "--data_types", help="List of data types to process", nargs="+", choices=["train", "val"])
    parser.add_argument("-i", "--iteration", type=int, default=0, help="The iteration number")
    parser.add_argument("-e", "--elogger", action="store_true", default=False, help="Whether to send email alerts")
    args = parser.parse_args()

    elogger.set_activate(args.elogger)

    # Load the config
    config = OmegaConf.load(args.config)

    # Get all the failed rollouts
    failed_rollout_paths = get_failed_rollouts(config, args.data_types, args.iteration)

    with open(config.leap.expert_correction_prompt, "r") as f:
        expert_correction_prompt = Template(f.read())

    with open(config.leap.summary_prompt, "r") as f:
        expert_summary_prompt = Template(f.read())

    print(f"======= Correcting {len(failed_rollout_paths)} failed rollouts")

    total_cost = 0
    # Correct the rollouts
    for failed_rollout_path, obj, category in tqdm(failed_rollout_paths, desc="Correcting rollouts"):
        print(f"\nCorrecting {failed_rollout_path}")
        total_cost += relabel_one_rollout(failed_rollout_path, expert_correction_prompt, expert_summary_prompt, obj, category)
        print(f"====== Total cost so far: ${total_cost:.2f} ======")

    elogger.log(f"Successfully corrected {len(failed_rollout_paths)} failed rollouts (total cost: ${total_cost:.2f})")