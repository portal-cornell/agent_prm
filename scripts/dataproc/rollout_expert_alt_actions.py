"""
Example usage:

When getting export's alternative actions:
    python scripts/dataproc/rollout_expert_alt_actions.py -e -m g -i 1 -d val -min 0 -max 4

    where
        -e indicates that we are using elogger
        -m g indicates that we are generating rollouts with alt actions
        -i 1 indicates the iteration that we are on (which affects the rollout directory)
        -d val indicates that we are processing the validation set.

When completing the rollouts:
    python scripts/dataproc/rollout_expert_alt_actions.py -e -m r -i 1 -d train --rollout_idx_min 4 --rollout_idx_max 12 -si 0 -ei 27 -s -p 40

    where
        -e indicates that we are using elogger
        -m r indicates that we are completing the rollouts
        -i 1 indicates the iteration that we are on (which affects the rollout directory and the agent config)
        -d train indicates that we are processing the training set
        --rollout_idx_min 4 --rollout_idx_max 12 indicates that we are completing the rollouts from idx 4 to 12
        -si 0 -ei 27 indicates that we are starting from object 0 and ending at object 27 (not inclusive)
        -s indicates that we are serving the model
        -p 40 indicates that we are using port 40

    python scripts/dataproc/rollout_expert_alt_actions.py -e -m g -i 1 -d train --curr_sample_iter 1 --starting_num_rollouts 4

When merging the summary dicts:
    python scripts/dataproc/rollout_expert_alt_actions.py -m merge_gen -i 0 -d train

    python scripts/dataproc/rollout_expert_alt_actions.py -m merge_rollout -i 0 -d val
"""
import argparse
import os
import random
import json
import math
import time
import copy
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

ALL_OBJ_LIST = [wv[0] for wv in get_default_word_list("all")]

iter_to_rollout_dir = {
    # pi0 3 epochs
    # 0: "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/hindsight_pi0-all-data-3epoches_250307_212417_peft=false_epoch3+all",
    0 : "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/hindsight-redo_pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all",
    # pi1-77pct_Q0-80pct-lr=5e-6-hindsight
    1: "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-77pct_Q0-80pct-lr=5e-6-hindsight_250321_225328_iter1_hindsight_pi0_Q0-80pct-lr=5e-6-hindsight"
}

iter_to_agent_config = {
    0: {
        "type": "sglang_server",
        "log_name": "pi0-all-data-3epoches",
        "model_id": "/share/portal/hw575/agent_prm/save/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120",
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
        "log_name": "pi1-77pct_Q0-80pct-lr=5e-6-hindsight",
        "model_id": "/share/portal/hw575/agent_prm/save/online_dpo/250321_225328_iter1_hindsight_pi0_Q0-80pct-lr=5e-6-hindsight/checkpoint-160",
        "prompt_template_file": "prompts/twenty_questions/twenty_questions_template.j2",
        "server_url": "http://localhost:TODO/",
        "dist_url_port": None,
        "temperature": 0.3,
        "batch_limit": 32,
        "verbose": 0,
        "debug": False,
    }
}

NUM_ALT_RESPONSES = 5

with open("prompts/twenty_questions/twenty_question_summary.j2", "r") as f:
    SUMMARY_PROMPT_TEMPLATE = Template(f.read())

with open("prompts/twenty_questions/twenty_question_expert_gen_prefered_action.j2", "r") as f:
    GEN_ACTION_PROMPT_TEMPLATE = Template(f.read())

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

def gen_alt_actions(
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
    

def generate_rollouts_with_alt_actions(
        rollout_dir: str, 
        n_rollouts_to_sample: int, 
        m_timesteps_to_gen_from: int, 
        actions_to_gen_at_each_timestep: int, 
        rollout_idx_min, 
        rollout_idx_max,
        data_types: List[str]=["train", "val"],
        start_obj_idx: int=-1, 
        end_obj_idx: int=-1):
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
                        rollout_idx += 1
                        continue

                    alt_reason_actions, cost = gen_alt_actions(obj, category, summary, rollout, t, actions_to_gen_at_each_timestep)

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
                        new_partial_rollout[t] = {
                            "step": t,
                            "teacher_reason": alt_reason_actions[i]["teacher_reason"],
                            "feasibility": alt_reason_actions[i]["feasibility"],
                            "reason": alt_reason_actions[i]["reason"],
                            "action": alt_reason_actions[i]["action"],
                            "raw_text": "",  # Because we are using gpt-4o, it has less parsing issues. 
                        }
                        new_partial_rollout = new_partial_rollout[:t+1]

                        # Save the new partial rollout
                        task_name = rollout_file.split("_")[0]

                        print(f"Saving new partial rollout: {os.path.join(rollout_dir, data_type, f'{task_name}_{rollout_idx}.json')}")

                        save_json(os.path.join(rollout_dir, data_type, f"{task_name}_{rollout_idx}.json"), new_partial_rollout)
                        if alt_reason_actions[i]["feasibility"] == "low":
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

    if args.mode == "g" or args.mode == "gen_actions":
        rollout_dir = iter_to_rollout_dir[args.iter]

        print(f"Generating rollouts with alt actions for data_type={args.data_types} in {rollout_dir}")

        try:
            generate_rollouts_with_alt_actions(
                rollout_dir=rollout_dir, 
                n_rollouts_to_sample=args.n_rollouts_to_sample, 
                m_timesteps_to_gen_from=args.m_timesteps_to_gen_from, 
                actions_to_gen_at_each_timestep=args.actions_to_gen_at_each_timestep, 
                rollout_idx_min=args.rollout_idx_min, 
                rollout_idx_max=args.rollout_idx_max,
                data_types=args.data_types, 
                start_obj_idx=args.start_obj_idx, 
                end_obj_idx=args.end_obj_idx)
        except Exception as e:
            elogger.log(f"Error generating rollouts with alt actions: {e}")
            raise e

        elogger.log(f"Successfully generated rollouts with alt actions for data_type={args.data_types} in {rollout_dir}")
    elif args.mode == "r" or args.mode == "rollout":
        rollout_dir = iter_to_rollout_dir[args.iter]
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
        rollout_dir = iter_to_rollout_dir[args.iter]
        merge_generation_summary_dicts(rollout_dir=rollout_dir, data_types=args.data_types)
    elif args.mode == "merge_rollout":
        rollout_dir = iter_to_rollout_dir[args.iter]
        merge_rollout_summary_dicts(rollout_dir=rollout_dir, data_types=args.data_types)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")