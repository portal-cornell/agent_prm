"""
Example usage:

When getting export's alternative actions:
    python scripts/dataproc/hindsight/rollout_expert_alt_actions_twenty_questions.py -e -m g -i 0 -d val -min 0 -max 4 -et gpt4o

    where
        -e indicates that we are using elogger
        -m g indicates that we are generating rollouts with alt actions
        -i 1 indicates the iteration that we are on (which affects the rollout directory)
        -d val indicates that we are processing the validation set.

When completing the rollouts:
    Without specifying the object range:
        python scripts/dataproc/hindsight/rollout_expert_alt_actions_twenty_questions.py -e -m r -i 0 -d train -min 4 -max 12 -et gpt4o --sim_host TODO --sim_port TODO

        where
            -e indicates that we are using elogger
            -m r indicates that we are completing the rollouts
            -i 1 indicates the iteration that we are on (which affects the rollout directory and the agent config)
            -d train indicates that we are processing the training set
            --min 4 --max 12 indicates that we are completing the rollouts from idx 4 to 12
            -s indicates that we are serving the model

    python scripts/dataproc/hindsight/rollout_expert_alt_actions_twenty_questions.py -e -m r -i 1 -d train -min 4 -max 12 -si 0 -ei 27 -et gpt4o --sim_host TODO --sim_port TODO
        where
            -si 0 -ei 27 indicates that we are starting from object 0 and ending at object 27 (not inclusive)

When merging the summary dicts:
    python scripts/dataproc/hindsight/rollout_expert_alt_actions_twenty_questions.py -m merge_gen -i 0 -d train -et gpt4o

    python scripts/dataproc/hindsight/rollout_expert_alt_actions_twenty_questions.py -m merge_rollout -i 0 -d val
"""
import argparse
import os
import random
import json
import math
import time
import copy
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Tuple
from jinja2 import Template

from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT, ALL_OBJECT_TO_CATEGORY, get_default_word_list, WordVariants
from agent_prm.envs.twenty_questions.interface import rollout_batch
from agent_prm.envs.twenty_questions.env import BatchedTwentyQuestionsEnvironment, setup_batched_twenty_questions_env
from agent_prm.agents.agent_registry import initialize_agent
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json, parse_reason_and_action_twenty_questions
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import start_sglang_server
from agent_prm.agents.agent import Agent

ALL_OBJ_LIST = [wv[0] for wv in get_default_word_list("all")]

iter_to_rollout_dir = {
    0 : {
        "gpt4o": "REDACTED",
        "pi*": "REDACTED",
        "explorative-pi": "REDACTED",
        "high-temp-pi": "REDACTED",
    },
    1: {
        "gpt4o": "REDACTED",
    },
    2: {
        "gpt4o": "REDACTED",
    }
}

iter_to_agent_config = {
    0: {
        "type": "sglang_server",
        "log_name": "pi0",
        "model_id": "REDACTED",
        "prompt_template_file": "prompts/twenty_questions/twenty_questions_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    },
    1: {
        "type": "sglang_server",
        "log_name": "pi1",
        "model_id": "REDACTED",
        "prompt_template_file": "prompts/twenty_questions/twenty_questions_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    },
    2: {
        "type": "sglang_server",
        "log_name": "pi2",
        "model_id": "REDACTED",
        "prompt_template_file": "prompts/twenty_questions/twenty_questions_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    }
}


high_temp_agent_config = {
    "type": "sglang_server",
    "log_name": "pi0",
    "model_id": "REDACTED",
    "prompt_template_file": "prompts/twenty_questions/twenty_questions_template.j2",
    "server_url": "http://localhost:TODO/",
    "dist_url_port": None,
    "temperature": 1.0,
    "batch_limit": 32,
    "verbose": 0,
    "debug": False,
}

best_agent_config = {
    "type": "sglang_server",
    "log_name": "pi2",
    "model_id": "REDACTED",
    "prompt_template_file": "prompts/twenty_questions/twenty_questions_template.j2",
    "server_url": "http://localhost:TODO/",
    "dist_url_port": None,
    "temperature": 0.7,
    "batch_limit": 32,
    "verbose": 0,
    "debug": False,
}

NUM_ALT_RESPONSES = 5

with open("prompts/twenty_questions/twenty_question_summary.j2", "r") as f:
    SUMMARY_PROMPT_TEMPLATE = Template(f.read())

with open("prompts/twenty_questions/twenty_question_expert_gen_prefered_action.j2", "r") as f:
    GEN_ACTION_PROMPT_TEMPLATE = Template(f.read())

with open("prompts/twenty_questions/twenty_questions_exploration_template.j2", "r") as f:
    EXPLORE_ACTION_PROMPT_TEMPLATE = Template(f.read())

with open("prompts/twenty_questions/twenty_questions_template.j2", "r") as f:
    OG_ACTION_PROMPT_TEMPLATE = Template(f.read())

def is_valid_rollout(f: str) -> bool:
    """
    Check if the rollout is valid
    """
    return f.endswith(".json") and not f.endswith("_original.json") and not "summary_dict" in f

def is_within_valid_range(file_name: str, rollout_idx_min: int, rollout_idx_max: int) -> bool:
    """
    Check if the rollout idx is within the valid range
    """
    if rollout_idx_min == -1 and rollout_idx_max == -1:
        return True

    rollout_idx = int(file_name.split("_")[-1].split(".")[0])

    return rollout_idx >= rollout_idx_min and rollout_idx < rollout_idx_max

def need_to_complete(file_name: str, summary_dict: Dict) -> bool:
    """
    Check if the rollout needs to be completed
    """
    obj = file_name.split("_")[0]
    rollout_idx = str(int(file_name.split("_")[-1].split(".")[0]))

    return rollout_idx not in summary_dict or obj not in summary_dict[rollout_idx]

def format_chat_history_and_goal(rollout: List[Dict], t: int, file_name: str="", category: str="") -> Tuple[str, str]:
    """
    Return
        - chat_history: str (until t)
        - goal: str (optional, if file_name is provided)
    """
    # Get the answer from the path name
    if file_name != "":
        answer = file_name.split("_")[0].lower()
        answer_str = f"The secret word is '{answer}'. It's in the general category '{category}'."
    else:
        answer_str = ""

    history_str = ""
    for i in range(t):
        history_str += f"Question #{i+1}: {rollout[i]['action']}\nAnswer #{i+1}: {rollout[i]['answer']}\n"

    return history_str, answer_str

def gen_summary_from_rollout(file_name: str, rollout: List[Dict], category: str) -> str:
    """
    Generate a summary from the rollout
    """
    chat_history, answer_str = format_chat_history_and_goal(rollout, len(rollout), file_name, category)

    system_prompt = SUMMARY_PROMPT_TEMPLATE.render(system=True, all_obj_list=ALL_OBJ_LIST)
    input_prompt = SUMMARY_PROMPT_TEMPLATE.render(system=False, mode="input", observation_action_history=chat_history, goal=answer_str)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(messages=messages, model="gpt-4o", temperature=0.3)

    return response, cost

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

def hindsight_gen_alt_actions(
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
    chat_history, _ = format_chat_history_and_goal(rollout, t)

    system_prompt = GEN_ACTION_PROMPT_TEMPLATE.render(system=True, all_obj_list=ALL_OBJ_LIST, summary=summary, num_responses=num_alt_actions_to_gen)

    input_prompt = GEN_ACTION_PROMPT_TEMPLATE.render(system=False, mode="input", observation_action_history=chat_history, num_responses=num_alt_actions_to_gen)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o", temperature=0.7
    )

    reason_action_list = parse_json(response)
    try:
        assert reason_action_list is not None, f"Failed to parse response: {response}"
        for reason_action in reason_action_list:
            assert "teacher_reason" in reason_action and "player_reason" in reason_action and "question" in reason_action, f"Invalid response: {reason_action}. Must contain 'teacher_reason', 'player_reason' and 'question'"
    except Exception as e:
        print(f"Error parsing response: {response}")
        raise e
    
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
            input("Press Enter to continue...")

    return reason_action_list, cost
    
def pi_gen_alt_actions(
    rollout: List[Dict],
    t: int,
    num_alt_actions_to_gen: int,
    agent: Agent
) -> List[Dict]:
    """
    Generate the alternative actions for the given timestep
    """
    history = [
        {
            "question": rollout[i]["action"],
            "answer": rollout[i]["answer"]
        } for i in range(t)
    ]

    if t == 19:
        input("stop")

    input_datas = [{
        "mode": "input",  # Based on how we sample the timesteps, we will never need to use the "input_final" mode
        "all_obj_list": ALL_OBJ_LIST,
        "observation_action_history": history,
    } for _ in range(num_alt_actions_to_gen)]

    reason_actions, _ = agent.predict_reason_action_batch(input_datas, num_responses=1) # List[List[Dict]]\

    # Flatten the list of lists
    reason_actions = [item for sublist in reason_actions for item in sublist]

    return reason_actions, 0

def format_common_actions(common_actions: List[str]) -> str:
    """
    Format the common actions to be used in the prompt
    """
    common_actions_str = ""
    for i, action in enumerate(common_actions):
        common_actions_str += f"Idea (You don't know if this is a good idea or not. You MUST NOT generate this question.): {action}\n"
    return common_actions_str.strip()

def explorative_pi_gen_alt_actions(
    rollout: List[Dict],
    t: int,
    num_alt_actions_to_gen: int,
    agent: Agent,
    common_actions_to_gen: int = 5
) -> List[Dict]:
    history = [
        {
            "question": rollout[i]["action"],
            "answer": rollout[i]["answer"]
        } for i in range(t)
    ]

    if t == 19:
        input("stop")

    input_datas = [{
        "mode": "input",  # Based on how we sample the timesteps, we will never need to use the "input_final" mode
        "all_obj_list": ALL_OBJ_LIST,
        "observation_action_history": history,
    } for _ in range(common_actions_to_gen)]

    agent.set_prompt_template(prompt_template=OG_ACTION_PROMPT_TEMPLATE)

    # Generate the common actions
    reason_actions, _ = agent.predict_reason_action_batch(input_datas, num_responses=1, alt_temperature_for_extra_responses=1.0) # List[List[Dict]]

    # Flatten the list of lists
    reason_actions = [item for sublist in reason_actions for item in sublist]

    # Collect the common actions
    common_actions = list(set([reason_action["action"] for reason_action in reason_actions]))

    # Change the prompt to generate the alternative actions
    agent.set_prompt_template(prompt_template=EXPLORE_ACTION_PROMPT_TEMPLATE)

    # Generate the alternative actions
    input_datas = [{
        "mode": "input",  # Based on how we sample the timesteps, we will never need to use the "input_final" mode
        "all_obj_list": ALL_OBJ_LIST,
        "observation_action_history": history,
        "common_questions": format_common_actions(common_actions)
    } for _ in range(num_alt_actions_to_gen)]

    reason_actions, _ = agent.predict_reason_action_batch(input_datas, num_responses=1, alt_temperature_for_extra_responses=1.0) # List[List[Dict]]

    # Flatten the list of lists
    reason_actions = [item for sublist in reason_actions for item in sublist]

    return reason_actions, 0

def high_temp_pi_gen_alt_actions(
    rollout: List[Dict],
    t: int,
    num_alt_actions_to_gen: int,
    agent: Agent,
    candidate_actions_to_gen: int = 15
) -> List[Dict]:
    history = [
        {
            "question": rollout[i]["action"],
            "answer": rollout[i]["answer"]
        } for i in range(t)
    ]

    if t == 19:
        input("stop")

    input_datas = [{
        "mode": "input",  # Based on how we sample the timesteps, we will never need to use the "input_final" mode
        "all_obj_list": ALL_OBJ_LIST,
        "observation_action_history": history,
    } for _ in range(candidate_actions_to_gen)]

    agent.set_prompt_template(prompt_template=OG_ACTION_PROMPT_TEMPLATE)

    # Generate the common actions
    reason_actions, _ = agent.predict_reason_action_batch(input_datas, num_responses=1, alt_temperature_for_extra_responses=1.0) # List[List[Dict]]

    # Flatten the list of lists
    reason_actions = [item for sublist in reason_actions for item in sublist]

    # Collect the common actions
    common_action_occurence_count = Counter([reason_action["action"] for reason_action in reason_actions])
    
    lack_enough_unique_action = len(common_action_occurence_count) < num_alt_actions_to_gen
    if lack_enough_unique_action:
        print(f"Lack enough unique actions. Only {len(common_action_occurence_count)} unique actions found.")

    # Rank the reason actions by the number of occurences
    reason_actions = sorted(reason_actions, key=lambda x: common_action_occurence_count[x["action"]], reverse=True)

    unique_reasons = []
    unique_actions = []
    for reason_action in reason_actions:
        if (lack_enough_unique_action or reason_action["action"] not in unique_actions) and len(unique_actions) < num_alt_actions_to_gen:
            unique_actions.append(reason_action["action"])
            unique_reasons.append(reason_action["reason"])

    assert len(unique_actions) == num_alt_actions_to_gen, f"Expected {num_alt_actions_to_gen} unique actions, but got {len(unique_actions)}"

    # Format unique_reasons and unique_actions into a List[Dict]
    unique_reason_actions = [{"reason": reason, "action": action} for reason, action in zip(unique_reasons, unique_actions)]

    return unique_reason_actions, 0

def generate_rollouts_with_alt_actions(
        rollout_dir: str, 
        n_rollouts_to_sample: int, 
        m_timesteps_to_gen_from: int, 
        actions_to_gen_at_each_timestep: int, 
        rollout_idx_min, 
        rollout_idx_max,
        expert_type: str,
        data_types: List[str]=["train", "val"],
        start_obj_idx: int=-1, 
        end_obj_idx: int=-1,
        agent: Agent=None):
    total_cost = 0

    for data_type in data_types:
        if data_type == "train":
            object_dict_to_use = TRAIN_OBJECT_DICT
        elif data_type == "val":
            object_dict_to_use = VALIDATION_OBJECT_DICT

        objects_to_eval_on = [(obj, category, data_type) for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]
        if start_obj_idx != -1:
            objects_to_eval_on = objects_to_eval_on[start_obj_idx:]
        if end_obj_idx != -1:
            objects_to_eval_on = objects_to_eval_on[:end_obj_idx]
        print(f"Evaluating on {len(objects_to_eval_on)} objects [{start_obj_idx},{end_obj_idx}): {objects_to_eval_on}")

        summary_dict_path = os.path.join(rollout_dir, data_type, f"_summary_dict_gen_range={rollout_idx_min}-{rollout_idx_max}_si={start_obj_idx}_ei={end_obj_idx}.json")
        if not os.path.exists(summary_dict_path):
            summary_dict = load_json(os.path.join(rollout_dir, data_type, "_summary_dict.json"))  # Assume that the summary dict is already generated
            save_json(summary_dict_path, summary_dict)

        # Get all the rollout files
        json_files = [f for f in os.listdir(os.path.join(rollout_dir, data_type)) if is_valid_rollout(f)]

        # For each task, we sample N rollouts
        for obj, category, _ in tqdm(objects_to_eval_on, desc="Processing objects"):
            # Get the task specific rollout files
            task_rollout_files = [f for f in json_files if f"{obj}_" in f and is_within_valid_range(f, rollout_idx_min, rollout_idx_max)]

            # Randomly select N rollouts (no replacement)
            selected_rollouts = random.sample(task_rollout_files, n_rollouts_to_sample)

            print(f"Selected rollouts: {selected_rollouts}")

            rollout_idx = rollout_idx_max

            # For each rollout, we sample M actions
            for rollout_file in selected_rollouts:
                rollout = load_json(os.path.join(rollout_dir, data_type, rollout_file))

                if expert_type == "gpt4o":
                    if "summary" not in rollout[0]:
                        summary, cost = gen_summary_from_rollout(rollout_file, rollout, category)
                        rollout[0]["summary"] = summary
                        save_json(os.path.join(rollout_dir, data_type, rollout_file), rollout)
                        total_cost += cost
                    else:
                        summary = rollout[0]["summary"]
                        total_cost += 0

                    print(f"Summary for {rollout_file} (cost: {total_cost:.2f}):\n{summary}")

                # Randomly sample M distinct timesteps (from 30% of the trajectory to 60% of the trajectory) to generate expert queries.
                start_timestep = math.floor(0.3 * len(rollout))
                end_timestep = math.ceil(0.6 * len(rollout))
                timestep_to_gen_from = random.sample(range(start_timestep, end_timestep), m_timesteps_to_gen_from)

                print(f"Rollout file: {rollout_file}, range: {range(start_timestep, end_timestep)}, Timesteps to generate from: {timestep_to_gen_from}")

                for t in timestep_to_gen_from:
                    # Check if the rollout has already been generated
                    summary_dict = load_json(summary_dict_path)
                    if str(rollout_idx) in summary_dict and obj in summary_dict[str(rollout_idx)]:
                        print(f"Rollout {rollout_idx} for {obj} has already been generated. Skipping...")
                        rollout_idx += 2 # Skipping 2 because we save 2 files per timestep
                        continue
                    
                    if expert_type == "gpt4o":
                        alt_reason_actions, cost = hindsight_gen_alt_actions(obj, category, summary, rollout, t, actions_to_gen_at_each_timestep)
                    elif expert_type == "pi*":
                        alt_reason_actions, cost = pi_gen_alt_actions(rollout, t, actions_to_gen_at_each_timestep, agent)
                    elif expert_type == "explorative-pi":
                        alt_reason_actions, cost = explorative_pi_gen_alt_actions(rollout, t, actions_to_gen_at_each_timestep, agent)
                    elif expert_type == "high-temp-pi":
                        alt_reason_actions, cost = high_temp_pi_gen_alt_actions(rollout, t, actions_to_gen_at_each_timestep, agent)

                    # Edit the rollout file to save the alt actions
                    if "expert_alternatives" not in rollout[t]:
                        rollout[t]["expert_alternatives"] = []

                    rollout[t]["expert_alternatives"].extend(alt_reason_actions)

                    save_json(os.path.join(rollout_dir, data_type, rollout_file), rollout)

                    # Make copy of the rollout file
                    for i in range(actions_to_gen_at_each_timestep):
                        new_partial_rollout = copy.deepcopy(rollout)
                        new_partial_rollout[0].pop("summary", None)  # Safely remove the summary

                        # Edit the action at t
                        if expert_type == "gpt4o":
                            new_partial_rollout[t] = {
                                "step": t,
                                "teacher_reason": alt_reason_actions[i]["teacher_reason"],
                                "feasibility": alt_reason_actions[i]["feasibility"],
                                "reason": alt_reason_actions[i]["reason"],
                                "action": alt_reason_actions[i]["action"],
                                "raw_text": "",  # Because we are using gpt-4o, it has less parsing issues. 
                            }
                        else:
                            new_partial_rollout[t] = {
                                "step": t,
                                "reason": alt_reason_actions[i]["reason"],
                                "action": alt_reason_actions[i]["action"],
                                "raw_text": "",  # This help signify that this action is generated by an expert
                            }
                        new_partial_rollout = new_partial_rollout[:t+1]

                        # Save the new partial rollout
                        task_name = rollout_file.split("_")[0]

                        print(f"Saving new partial rollout: {os.path.join(rollout_dir, data_type, f'{task_name}_{rollout_idx}.json')}")

                        save_json(os.path.join(rollout_dir, data_type, f"{task_name}_{rollout_idx}.json"), new_partial_rollout)

                        if expert_type == "gpt4o" and alt_reason_actions[i]["feasibility"] == "low":
                            input("Press Enter to continue...")

                        # Update the summary dict
                        summary_dict = load_json(summary_dict_path)
                        if str(rollout_idx) not in summary_dict:
                            summary_dict[str(rollout_idx)] = []
                        if obj not in summary_dict[str(rollout_idx)]:
                            summary_dict[str(rollout_idx)].append(obj)
                        save_json(summary_dict_path, summary_dict)

                        rollout_idx += 1

                    print(f"Total cost: {total_cost:.2f}")
                    # input("Press Enter to continue...")



def prepare_batch(batch_json_files: List[Tuple[str, str, str]], batched_env: BatchedTwentyQuestionsEnvironment):
    """
    Return:
        - histories: List[List[Dict]]
        - words_to_guess: List[WordVariants]
        - traj_list: List[List[Dict]]
        - prev_dones: List[bool]
    """
    histories = []
    actions = []
    traj_list = []
    categories = []
    for file, data_type, category in batch_json_files:
        rollout = load_json(os.path.join(rollout_dir, data_type, file))
        traj_list.append(rollout)
        actions.append(rollout[-1]["action"])  # The last action is the expert's action
        categories.append(category) # The secret word's category

        rollout_histories = [{
            "question": rollout[i]["action"],
            "answer": rollout[i]["answer"]
        } for i in range(len(rollout)-1)]  # -1 because the last action doesn't have an answer yet

        histories.append(rollout_histories)

    words_to_guess = [WordVariants.from_str(file.split("_")[0]) for file, _, _ in batch_json_files]

    prev_dones = [False for _ in range(len(histories))]
    # Take a step in the environment using expert's action
    histories, answer_reasons, answers, rewards, dones = batched_env.step(words_to_guess, categories, histories, actions, prev_dones)

    # Log the trajectories
    for i in range(len(batch_json_files)):
        if not prev_dones[i]:
            traj_list[i][-1]["answerer_reason"] = answer_reasons[i]
            traj_list[i][-1]["answer"] = answers[i]
            traj_list[i][-1]["reward"] = rewards[i]
            traj_list[i][-1]["score"] = None  # Critic is not used to score the expert's action
            traj_list[i][-1]["alternatives"] = None  # No alternative actions for the expert's action

    prev_dones = dones

    return histories, words_to_guess, categories, traj_list, prev_dones


def complete_rollouts(sim_host: str, sim_port: int, rollout_dir: str, agent_config: Dict, bs: int, rollout_idx_min: int=-1, rollout_idx_max: int=-1, data_types: List[str]=["train", "val"], start_obj_idx: int=-1, end_obj_idx: int=-1):
    """
    Complete the rollouts
    """
    # Initialize the agent
    print(f"Initializing agent {agent_config['log_name']}, {agent_config['model_id']}")
    agent = initialize_agent(agent_config,
                            parse_reason_action_fn=parse_reason_and_action_twenty_questions,
                            verbose=agent_config["verbose"],
                            debug=agent_config["debug"])
    batched_env = setup_batched_twenty_questions_env(host=sim_host,
                                                     port=sim_port)

    # Collect all the rollouts that need to be completed
    json_files_to_complete = []

    for data_type in data_types:
        summary_dict = load_json(os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json"))

        json_files = [(f, data_type, ALL_OBJECT_TO_CATEGORY[f.split("_")[0]]) for f in os.listdir(os.path.join(rollout_dir, data_type)) if is_valid_rollout(f) and need_to_complete(f, summary_dict) and is_within_valid_range(f, rollout_idx_min, rollout_idx_max)]

        if start_obj_idx != -1 or end_obj_idx != -1:
            if data_type == "train":
                object_dict_to_use = TRAIN_OBJECT_DICT
            elif data_type == "val":
                object_dict_to_use = VALIDATION_OBJECT_DICT

            objects_to_eval_on = [obj for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]
            if start_obj_idx != -1:
                objects_to_eval_on = objects_to_eval_on[start_obj_idx:]
            if end_obj_idx != -1:
                objects_to_eval_on = objects_to_eval_on[:end_obj_idx]

            json_files = [(f, data_type, category) for f, data_type, category in json_files if f.split("_")[0] in objects_to_eval_on]
            print(f"Filtering json files to {len(json_files)} rollouts range [{start_obj_idx},{end_obj_idx}): {objects_to_eval_on}")

        json_files_to_complete.extend(json_files)

        # Save a copy of the summary dict
        summary_dict_path = os.path.join(rollout_dir, data_type, f"_rollout_complete_summary_dict_range={rollout_idx_min}-{rollout_idx_max}_si={start_obj_idx}_ei={end_obj_idx}.json")
        if not os.path.exists(summary_dict_path):
            save_json(summary_dict_path, summary_dict)
        else:
            summary_dict = load_json(summary_dict_path)

    # Filter out the rollouts that are already completed
    filtered_json_files_to_complete = []
    for file, data_type, category in json_files_to_complete:
        rollout_idx = file.split("_")[-1].split(".")[0]
        obj = file.split("_")[0]
        if rollout_idx not in summary_dict or obj not in summary_dict[rollout_idx]:
            filtered_json_files_to_complete.append((file, data_type, category))
    json_files_to_complete = filtered_json_files_to_complete

    print(f"Total number of rollouts to complete: {json_files_to_complete}\nlen: {len(json_files_to_complete)}")
    # input("stop")

    for batch in tqdm(range(math.ceil(len(json_files_to_complete) / bs))):
        batch_json_files = json_files_to_complete[batch * bs:(batch + 1) * bs]

        print(f"Batch {batch} has {len(batch_json_files)} rollouts: {batch_json_files}")

        # Prepare the batch
        histories, words_to_guess, categories, traj_list, prev_dones = prepare_batch(batch_json_files, batched_env)

        # Rollout the batch
        traj_list = rollout_batch(agent, batched_env, ALL_OBJ_LIST, words_to_guess, categories, histories, traj_list, prev_dones, NUM_ALT_RESPONSES)

        # Save the trajectories
        for i in range(len(batch_json_files)):
            data_type = batch_json_files[i][1]
            file = batch_json_files[i][0]
            save_json(os.path.join(rollout_dir, data_type, file), traj_list[i])

            obj = file.split("_")[0]
            rollout_idx = str(int(file.split("_")[-1].split(".")[0]))

            summary_dict = load_json(summary_dict_path)
            if rollout_idx not in summary_dict:
                summary_dict[rollout_idx] = []

            summary_dict[rollout_idx].append(obj)
            save_json(summary_dict_path, summary_dict)

def merge_rollout_summary_dicts(rollout_dir: str, data_types: List[str]) -> Dict:
    """
    Merge the summary dicts
    """
    for data_type in data_types:
        original_rollout_summary_dict_path = os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json")
        rollout_summary_dicts_path = [f for f in os.listdir(os.path.join(rollout_dir, data_type)) if "_rollout_complete_summary_dict" in f and  "_rollout_complete_summary_dict.json" not in f and "copy" not in f]
        rollout_summary_dicts_path.sort()
        print(f"Rollout summary dicts path: {rollout_summary_dicts_path}")
        input("stop")
        
        main_summary_dict = load_json(original_rollout_summary_dict_path)

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            summary_dict = load_json(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))

            for rollout_idx, obj_list in summary_dict.items():
                if rollout_idx not in main_summary_dict:
                    main_summary_dict[rollout_idx] = []

                for obj in obj_list:
                    if obj not in main_summary_dict[rollout_idx]:
                        main_summary_dict[rollout_idx].append(obj)

        # Sort the summary dict by the rollout idx
        main_summary_dict = dict(sorted(main_summary_dict.items(), key=lambda x: int(x[0])))

        print(f"Main summary dict keys: {main_summary_dict.keys()}")
        print(f"Main summary value len: {[len(v) for v in main_summary_dict.values()]}")
        input("Check the merged summary dict before we save it")
        
        save_json(os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json"), main_summary_dict)

        input("Please check the merged summary dict before we delete the intermediate summary dicts")

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            print(f"Deleting {os.path.join(rollout_dir, data_type, rollout_summary_dict_path)}")
            input("stop")
            os.remove(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))


def merge_generation_summary_dicts(rollout_dir: str, data_types: List[str]) -> Dict:
    """
    Merge the summary dicts
    """
    for data_type in data_types:
        original_rollout_summary_dict_path = os.path.join(rollout_dir, data_type, "_summary_dict.json")
        rollout_summary_dicts_path = [f for f in os.listdir(os.path.join(rollout_dir, data_type)) if "_summary_dict" in f and  "rollout_complete" not in f and "copy" not in f and "_summary_dict.json" not in f]
        rollout_summary_dicts_path.sort()
        print(f"Rollout summary dicts path: {rollout_summary_dicts_path}")
        input("stop")
        
        main_summary_dict = load_json(original_rollout_summary_dict_path)
        # Save a copy for rollout_complete
        save_json(os.path.join(rollout_dir, data_type, "_rollout_complete_summary_dict.json"), main_summary_dict)
        input("Saved the rollout complete summary dict")

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            summary_dict = load_json(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))

            for rollout_idx, obj_list in summary_dict.items():
                if rollout_idx not in main_summary_dict:
                    main_summary_dict[rollout_idx] = []

                for obj in obj_list:
                    if obj not in main_summary_dict[rollout_idx]:
                        main_summary_dict[rollout_idx].append(obj)

        # Sort the summary dict by the rollout idx
        main_summary_dict = dict(sorted(main_summary_dict.items(), key=lambda x: int(x[0])))

        print(f"Main summary dict keys: {main_summary_dict.keys()}")
        print(f"Main summary value len: {[len(v) for v in main_summary_dict.values()]}")
        input("Check the merged summary dict before we save it")
        
        save_json(os.path.join(rollout_dir, data_type, "_summary_dict.json"), main_summary_dict)

        input("Please check the merged summary dict before we delete the intermediate summary dicts")

        for rollout_summary_dict_path in rollout_summary_dicts_path:
            print(f"Deleting {os.path.join(rollout_dir, data_type, rollout_summary_dict_path)}")
            input("stop")
            os.remove(os.path.join(rollout_dir, data_type, rollout_summary_dict_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, choices=list(range(3)), required=True)
    parser.add_argument("-m", "--mode", type=str, choices=["g", "gen_actions", "r", "rollout", "merge_gen", "merge_rollout"], required=True)
    parser.add_argument("-et", "--expert_type", type=str, choices=["gpt4o", "pi*", "explorative-pi", "high-temp-pi"], required=True)
    parser.add_argument("-d", "--data_types", help="List of data types to process", nargs="+", choices=["train", "val"])
    parser.add_argument("-e", "--elogger", action="store_true", default=False)
    parser.add_argument("--n_rollouts_to_sample", type=int, default=2)
    parser.add_argument("--m_timesteps_to_gen_from", type=int, default=2)
    parser.add_argument("--actions_to_gen_at_each_timestep", type=int, default=2)
    parser.add_argument("--starting_num_rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-min", "--rollout_idx_min", type=int, default=-1, help="When it's generating rollouts, define the range of rollout idx to sample from. When it's completing rollouts, define the range of rollout idx to complete")
    parser.add_argument("-max", "--rollout_idx_max", type=int, default=-1)
    parser.add_argument("-ns", "--no_serve_model", action="store_true", default=False)
    parser.add_argument("-p", "--port", type=str, default=None)
    parser.add_argument("-si", "--start_obj_idx", type=int, default=-1, help="Within a data_type, start from this object idx")
    parser.add_argument("-ei", "--end_obj_idx", type=int, default=-1, help="Within a data_type, end at this object idx")
    parser.add_argument("--sim_host", type=str, default="localhost")
    parser.add_argument("--sim_port", type=int, default=40042)
    args = parser.parse_args()
    random.seed(args.seed)

    elogger.set_activate(args.elogger)

    rollout_dir = iter_to_rollout_dir[args.iter][args.expert_type]
    if args.mode == "g" or args.mode == "gen_actions":
        print(f"Generating rollouts with alt actions for data_type={args.data_types} in {rollout_dir}")

        if args.expert_type == "pi*" or args.expert_type == "explorative-pi" or args.expert_type == "high-temp-pi":
            if args.expert_type == "high-temp-pi":
                agent_config = high_temp_agent_config
            elif args.expert_type == "pi*":
                agent_config = best_agent_config
            else:
                agent_config = iter_to_agent_config[args.iter]
            if not args.no_serve_model:
                # Example of dist_url_port: 29540
                process, server_url, base_gpu_id = start_sglang_server(model_path=agent_config["model_id"],
                                                        port=args.port, 
                                                        tp=1,
                                                        dist_url_port=agent_config["dist_url_port"])
                agent_config["server_url"] = server_url

            # Initialize the agent
            print(f"Initializing agent {agent_config['log_name']}, {agent_config['model_id']}")
            agent = initialize_agent(agent_config,
                                    parse_reason_action_fn=parse_reason_and_action_twenty_questions,
                                    verbose=agent_config["verbose"],
                                    debug=agent_config["debug"])

        try:
            generate_rollouts_with_alt_actions(
                rollout_dir=rollout_dir, 
                n_rollouts_to_sample=args.n_rollouts_to_sample, 
                m_timesteps_to_gen_from=args.m_timesteps_to_gen_from, 
                actions_to_gen_at_each_timestep=args.actions_to_gen_at_each_timestep, 
                rollout_idx_min=args.rollout_idx_min, 
                rollout_idx_max=args.rollout_idx_max,
                expert_type=args.expert_type,
                data_types=args.data_types, 
                start_obj_idx=args.start_obj_idx, 
                end_obj_idx=args.end_obj_idx,
                agent=agent)
        except Exception as e:
            elogger.log(f"Error generating rollouts with alt actions: {e}")
            raise e

        elogger.log(f"Successfully generated rollouts with alt actions for data_type={args.data_types} in {rollout_dir}")
    elif args.mode == "r" or args.mode == "rollout":
        agent_config = iter_to_agent_config[args.iter]

        if not args.no_serve_model:
            # Example of dist_url_port: 29540
            process, server_url, base_gpu_id = start_sglang_server(model_path=agent_config["model_id"],
                                                    port=args.port, 
                                                    tp=1,
                                                    dist_url_port=agent_config["dist_url_port"])
            agent_config["server_url"] = server_url

        complete_rollouts(sim_host=args.sim_host, sim_port=args.sim_port, rollout_dir=rollout_dir, agent_config=agent_config, bs=agent_config["batch_limit"], rollout_idx_min=args.rollout_idx_min, rollout_idx_max=args.rollout_idx_max, data_types=args.data_types, start_obj_idx=args.start_obj_idx, end_obj_idx=args.end_obj_idx)

        elogger.log(f"Successfully completed rollouts for data_type={args.data_types} in {rollout_dir}")
    elif args.mode == "merge_gen":
        merge_generation_summary_dicts(rollout_dir=rollout_dir, data_types=args.data_types)
    elif args.mode == "merge_rollout":
        merge_rollout_summary_dicts(rollout_dir=rollout_dir, data_types=args.data_types)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")