import os
import json
import argparse
import yaml
import random
from hashlib import sha256
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from functools import partial
from omegaconf import DictConfig, OmegaConf
import hydra
from typing import List, Dict

from agent_prm.utils.general_utils import load_json

"""================================================================================
    Alfworld processing functions
================================================================================"""

def alfworld_process_file(file_path, gamma, exclude_reason=False, is_offpolicy=False):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        if 'trajectory' not in data:
            return {}

        Q_target = {}
        trajectory = data['trajectory']
        outcome_reward = 2 * trajectory[-1]['score'] - 1  # transform from [-1, 1]
        for t in range(len(trajectory) - 1, -1, -1):
            state, reason_action = alfworld_extract_state_reason_action(trajectory, data['task'], t, exclude_reason=exclude_reason)
            state_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
            update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, t)
        return Q_target
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}
    
def alfworld_extract_state_reason_action(trajectory, task, t, exclude_reason=False):
    history = []
    for i in range(t):
        step = trajectory[i]
        history.append({
            'observation': step['observation'],
            'action': step['action']
        })
    
    state = {
        'observation': trajectory[t]['observation'],
        'candidate_actions': trajectory[t]['candidate_actions'],
        'history': history,
        'task': task
    }

    if exclude_reason:
        reason_action = {
            'action': trajectory[t]['action'],
        }
    else:
        reason_action = {
            'reason': trajectory[t]['reason'],
            'action': trajectory[t]['action'],
        }

    return state, reason_action

def alfworld_skip_file_condition(rolloutdir, file_name, max_rollout_per_task_per_dir=None, include_only_success=False):
    """
    For now, cannot handle include_only_success
    """
    return file_name.endswith(".json") or (max_rollout_per_task_per_dir is not None and int(file_name.split("_")[-1].split(".")[0]) >= max_rollout_per_task_per_dir)

def alfworld_success_file_condition(file_name):
    # TODO: Implement this
    return True

"""================================================================================
    20 Questions processing functions
================================================================================"""
def twenty_questions_process_file(file_path, gamma, exclude_reason=False, is_offpolicy=False):
    try:
        with open(file_path, 'r') as file:
            trajectory = json.load(file)

        Q_target = {}
        outcome_reward = twenty_questions_normalize_reward(trajectory[-1]['reward'])  # transform from [-1, 1]
        for t in range(len(trajectory) - 1, -1, -1):
            state, reason_action = twenty_questions_extract_state_reason_action(trajectory, t, exclude_reason=exclude_reason)
            state_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
            update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, k=t, T=len(trajectory), is_offpolicy=is_offpolicy)
        return Q_target
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

def twenty_questions_extract_state_reason_action(trajectory, t, exclude_reason=False):
    history = []
    for i in range(t-1):
        step = trajectory[i]
        history.append({
            'question': step['action'],
            'answer': step['answer']
        })
    
    state = {
        'history': history,
    }

    if exclude_reason:
        reason_action = {
            'action': trajectory[t]['action'],
        }
    else:
        reason_action = {
            'reason': trajectory[t]['reason'],
            'action': trajectory[t]['action'],
        }

    return state, reason_action
    
def twenty_questions_normalize_reward(reward):
    # normalize outcome reward from [-1, 0] to [-1, 1]
    #   We normalize the outcome reward to -1 and 1 so that we can proprogate the negative effect of failed trajectories
    #   We assume that the reward before the last step is still 0
    return 2 * reward + 1

def twenty_questions_skip_file_condition(rolloutdir, file_name, max_rollout_per_task_per_dir=None):
    return (not file_name.endswith(".json")) or ('_summary_dict' in file_name) or ('original' in file_name) or (max_rollout_per_task_per_dir is not None and int(file_name.split("_")[-1].split(".")[0]) >= max_rollout_per_task_per_dir)

def twenty_questions_filter_condition_checker(rolloutdir, file_name):
    """
    Return
        - True if the file failed
        - True if the file is an expert rollout that led to success
    """
    rollout = load_json(os.path.join(rolloutdir, file_name))

    is_expert_rollout = False
    idx_of_expert_action = 0
    for t in range(len(rollout)):
        if rollout[t]['raw_text'] == "":
            is_expert_rollout = True
            idx_of_expert_action = t

    failed = rollout[-1]['reward'] != 0
    if is_expert_rollout:
        # Filtering out the following cases:
        # 1. The expert rollout failed
        # 2. The expert action led to success
        expert_action_led_to_success = rollout[idx_of_expert_action]['reward'] == 0

        return failed, expert_action_led_to_success
    else:
        return failed, False
    

def twenty_questions_success_file_condition(file_name):
    # Read the file and check the last reward is 0
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data[-1]['reward'] == 0

"""================================================================================
    General functions shared by all tasks
================================================================================"""

def print_histogram(val_list, name, bins):
    counts, bin_edges = np.histogram(val_list, bins=bins)

    print(f"========== {name} histogram ==========")
    for i in range(len(counts)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {counts[i]}")
    print("=====================================")

def print_qestimate_histogram(Q_target, bins=10):
    qestimates = [entry['qestimate'] for entry in Q_target.values()]
    print_histogram(qestimates, "Q-estimate", bins)

def print_count_histogram(Q_target, bins):
    counts = [entry['count'] for entry in Q_target.values()]
    print_histogram(counts, "Count", bins)

def update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, k, T, is_offpolicy=False):
    """
    Update the Q estimate for a given state-action pair.

    Parameters:
        state: The state of the environment.
        reason_action: The action and the reasoning behind it.
        state_hash: The hash of the state.
        Q_target: The dictionary of Q estimates.
        outcome_reward: The reward for the action.
        gamma: The discount factor.
        t: The time step.
        T: The total number of time steps in this trajectory.
    """
    if state_hash not in Q_target:
        Q_target[state_hash] = {
            'state': state,
            'reason_action': reason_action,
            'qestimate': 0,
            'count': 0,
            'contains_offpolicy': is_offpolicy # Contains contribution from the off-policy rollouts
        }
    current_entry = Q_target[state_hash]
    current_qestimate = current_entry['qestimate']
    current_count = current_entry['count']
    
    # The Q estimate equation is:
    #   for each trajectory that passes through this state-action pair,
    #       calculate the discounted reward from time t to the end of the trajectory (k = t to k = T-1) sum(gamma^{k-t} * r_k)
    #   then divide by the number of trajectories that pass through this state-action pair
    # current_qestimate * current_count recovers the previous sum

    # If we assume that the reward is 0 until it succeeds and terminates, then the equation: (k = t to k = T-1) sum(gamma^{k-t} * r_k)
    # is equivalent to gamma^{T-1-t} * outcome_reward
    # print(f"previous qestimate: {current_qestimate}, previous count: {current_count}, sum: {current_qestimate * current_count}")
    # print(f"k={k}, gamma^k*outcome_reward: {gamma ** k * outcome_reward}")
    # print(f"k={k}, T={T}, gamma^{T-1-k}: {gamma ** (T-1-k) * outcome_reward}, outcome_reward: {outcome_reward}")
    # input("stop")
    updated_qestimate = (current_qestimate * current_count + (gamma ** (T-k-1)) * outcome_reward) / (current_count + 1)
    Q_target[state_hash]['qestimate'] = updated_qestimate
    Q_target[state_hash]['count'] = current_count + 1

    if is_offpolicy:
        Q_target[state_hash]['contains_offpolicy'] = True


# Function to merge results from multiple processes
def merge_results(results):
    merged_Q_target = {}
    for Q_target in results:
        for key, value in Q_target.items():
            if key not in merged_Q_target:
                merged_Q_target[key] = value
            else:
                # Merge results for duplicate keys
                current_entry = merged_Q_target[key]
                new_qestimate = (current_entry['qestimate'] * current_entry['count'] + 
                                 value['qestimate'] * value['count']) / (current_entry['count'] + value['count'])
                merged_Q_target[key]['qestimate'] = new_qestimate
                merged_Q_target[key]['count'] += value['count']
                merged_Q_target[key]['contains_offpolicy'] = merged_Q_target[key]['contains_offpolicy'] or value['contains_offpolicy']
    return merged_Q_target

def subsample_data(data: List[Dict], count_to_reduce: int, bins: int = 5, low_or_high: str = "low"):
    """
    Parameters:
        data: List of dictionaries, each containing 'qestimate' and 'state' and 'reason_action'
        count_to_reduce: The number of datapoints to remove
        bins: The number of bins to use for the histogram

    Returns:
        - The subsampled data
        - The number of datapoints left to reduce
    """
    if count_to_reduce == 0:
        return data, 0
    elif len(data) > count_to_reduce:
        q_val_list = [entry['qestimate'] for entry in data]

        num_datapoints_to_reduce = count_to_reduce
        print(f"num_datapoints_to_reduce: {num_datapoints_to_reduce}")

        # list of bins (0, 1/bins), (1/bins, 2/bins), ..., (1-1/bins, 1)
        if low_or_high == "low":
            bins_list = np.linspace(0, 0.5, bins+1)
        else:
            bins_list = np.linspace(0.5, 1, bins+1)
        
        # Find the bin that has the highest count of datapoints
        counts, bin_edges = np.histogram(q_val_list, bins=bins_list)
        print("========== Initial histogram ==========")
        print(f"counts: {counts}")
        print(f"bin_edges: {bin_edges}")
        print("==========================================")

        # Subsample the data in the bin with the highest count of datapoints until we have count datapoints
        new_data = []
        data_to_process = data

        total_removal_count = 0

        while total_removal_count < num_datapoints_to_reduce and (not all(counts == 0)):
            sorted_indices = np.argsort(-counts)  # Sort in descending order
            highest_bin_index = sorted_indices[0]
            third_highest_bin_index = sorted_indices[1]  # Assume there's always at least 3 bins

            # Calculate the number of datapoints to remove from the highest bin
            #   Remove datapoints from the highest bin until the second highest bin becomes the new highest bin
            removal_count = min(counts[highest_bin_index]-counts[third_highest_bin_index]+1, num_datapoints_to_reduce - total_removal_count)
            
            if (num_datapoints_to_reduce - total_removal_count)/removal_count > 10:
                # Manually increase the removal count to be more aggressive
                #   This is to ensure that we don't remove too many datapoints from the highest bin
                removal_count *= int((num_datapoints_to_reduce - total_removal_count)/removal_count/10)

            bin_to_remove_min = bin_edges[highest_bin_index]
            bin_to_remove_max = bin_edges[highest_bin_index + 1]

            print(f"Removing {removal_count} datapoints from bin {bin_to_remove_min} to {bin_to_remove_max}")
            
            count = 0
            new_data = []
            for entry in data_to_process:
                if entry['qestimate'] >= bin_to_remove_min and entry['qestimate'] <= bin_to_remove_max and count < removal_count:
                    # Removing the datapoint
                    count += 1
                else:
                    new_data.append(entry)

            total_removal_count += count
            
            # Recalculate the histogram
            new_q_val_list = [entry['qestimate'] for entry in new_data]
            counts, bin_edges = np.histogram(new_q_val_list, bins=bins_list)
            print(f"================ Removed {count} datapoints from bin {bin_to_remove_min} to {bin_to_remove_max}, {num_datapoints_to_reduce - total_removal_count} remaining =================")
            print(f"counts: {counts}")
            
            data_to_process = new_data
        
        print_histogram([entry['qestimate'] for entry in new_data], f"Final Q-estimate for {low_or_high} data (len={len(new_data)})", bins_list)

        return new_data, num_datapoints_to_reduce - total_removal_count
    else:
        count_to_reduce_left = count_to_reduce - len(data)
        return [], count_to_reduce_left

    
# Main function using multiprocessing
def compute_prm_target(files, files_breakdown, domain, outputdir, gamma, cpu_count=None, train_split=None, split_name=None, balance_data=False, onpolicy_pct_for_success=None, track_offpolicy=False):
    if domain == "alfworld":
        process_file = alfworld_process_file
    elif domain == "twenty_questions":
        process_file = twenty_questions_process_file
    else:
        raise ValueError(f"Invalid domain: {domain}")

    Q_target = {}

    # Automatically detect the number of CPUs
    if cpu_count is None:
        num_cpus = cpu_count()
    else:
        num_cpus = cpu_count
    print(f"Using {num_cpus} CPUs for parallel processing")

    # Use multiprocessing Pool for parallel processing
    if track_offpolicy:
        assert len(files_breakdown) == 4
        num_cpus_per_type = num_cpus // 4
        onpolicy_rollouts_failed, onpolicy_rollouts_succeeded, offpolicy_files_failed_to_include, offpolicy_files_good = files_breakdown

        results = []

        with Pool(processes=num_cpus_per_type) as pool:
            process_func = partial(process_file, gamma=gamma, is_offpolicy=False)
            results.extend(list(tqdm(pool.imap(process_func, onpolicy_rollouts_failed), total=len(onpolicy_rollouts_failed), desc="Processing onpolicy rollouts failed")))

        with Pool(processes=num_cpus_per_type) as pool:
            process_func = partial(process_file, gamma=gamma, is_offpolicy=False)
            results.extend(list(tqdm(pool.imap(process_func, onpolicy_rollouts_succeeded), total=len(onpolicy_rollouts_succeeded), desc="Processing onpolicy rollouts succeeded")))

        with Pool(processes=num_cpus_per_type) as pool:
            process_func = partial(process_file, gamma=gamma, is_offpolicy=True)
            results.extend(list(tqdm(pool.imap(process_func, offpolicy_files_failed_to_include), total=len(offpolicy_files_failed_to_include), desc="Processing offpolicy files failed to include")))
        
        with Pool(processes=num_cpus_per_type) as pool:
            process_func = partial(process_file, gamma=gamma, is_offpolicy=True)
            results.extend(list(tqdm(pool.imap(process_func, offpolicy_files_good), total=len(offpolicy_files_good), desc="Processing offpolicy files good")))
    else:
        with Pool(processes=num_cpus) as pool:
            process_func = partial(process_file, gamma=gamma)
            results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing files"))
    
    # # First test with single process
    # results = [process_file(file, gamma) for file in files]
    
    # Merge results from all processes
    Q_target = merge_results(results)

    # Transform Q_target back to [0, 1]
    for key in Q_target.keys():
        Q_target[key]['qestimate'] = 0.5 * (Q_target[key]['qestimate'] + 1)

    print_qestimate_histogram(Q_target)
    # print_count_histogram(Q_target, bins=np.array(list(range(1, 6)) + list(range(6, 10, 2)) +list(range(10, 100, 10)) + list(range(100, max([x['count'] for x in Q_target.values()]), 100))))
    if track_offpolicy:
        Q_target_onpolicy = {k: v for k, v in Q_target.items() if not v['contains_offpolicy']}
        Q_target_off_policy = {k: v for k, v in Q_target.items() if v['contains_offpolicy']}
        print("======= On-policy Q-estimate =======")
        print_qestimate_histogram(Q_target_onpolicy)
        print("======= Off-policy (Hindsight) Q-estimate =======")
        print_qestimate_histogram(Q_target_off_policy)
    input("Press any key to continue...")

    if split_name is not None:
        if not track_offpolicy:
            reduce_and_save_data_for_split(Q_target, outputdir, split_name, balance_data)
        else:
            reduce_and_save_data_for_split_biased(Q_target, outputdir, split_name, balance_data, onpolicy_pct_for_success)
    else:
        keys = list(Q_target.keys())
        random.shuffle(keys)

        # Split into train/val and save
        split_idx = int(len(keys) * train_split)
        train_keys, val_keys = keys[:split_idx], keys[split_idx:]

        train_data = [
            {'state': Q_target[k]['state'], 
            'reason_action': Q_target[k]['reason_action'], 
            'qestimate': Q_target[k]['qestimate']}
            for k in train_keys
        ]
        val_data = [
            {'state': Q_target[k]['state'], 
             'reason_action': Q_target[k]['reason_action'], 
             'qestimate': Q_target[k]['qestimate']}
            for k in val_keys
        ]

        # Convert lists of dictionaries to Arrow tables
        train_table = pa.Table.from_pylist(train_data)
        val_table = pa.Table.from_pylist(val_data)

        # Save the Arrow tables to Parquet files
        os.makedirs(outputdir, exist_ok=True)
        train_path = os.path.join(outputdir, 'train.parquet')
        val_path = os.path.join(outputdir, 'val.parquet')

        pq.write_table(train_table, train_path)
        pq.write_table(val_table, val_path)

        # Subsample the data to strictly having 10k datapoints
        train_table_10k = train_table.slice(0, 10000)
        pq.write_table(train_table_10k, os.path.join(outputdir, 'train_10k.parquet'))

def reduce_and_save_data_for_split(Q_target, outputdir, split_name, balance_data=False):
    keys = list(Q_target.keys())
    random.shuffle(keys)

    original_data_to_save = [
        {'state': Q_target[k]['state'], 
        'reason_action': Q_target[k]['reason_action'], 
        'qestimate': Q_target[k]['qestimate']}
        for k in keys
    ]

    if split_name == "val":
        if balance_data:
            # Find the data that has Q-estimate >= 0.5
            low_data = [x for x in original_data_to_save if x['qestimate'] < 0.5]
            high_data = [x for x in original_data_to_save if x['qestimate'] >= 0.5]

            count = min(len(low_data), len(high_data))

            # Subsample the data
            low_data_subsampled = subsample_data(low_data, count, low_or_high="low")
            high_data_subsampled = subsample_data(high_data, count, low_or_high="high")

            print(f"Reducing low_data from {len(low_data)} to {count}")
            print(f"Reducing high_data from {len(high_data)} to {count}")

            input("Press any key to continue...")

            data_to_save = low_data_subsampled + high_data_subsampled
        else:
            data_to_save = original_data_to_save

        data_table = pa.Table.from_pylist(data_to_save)
        
        # Save the Arrow tables to Parquet files
        os.makedirs(outputdir, exist_ok=True)
        pq.write_table(data_table, os.path.join(outputdir, f"{split_name}.parquet"))
        
        print(f"Saving {len(data_to_save)} datapoints as {split_name}.parquet")
    elif split_name == "train":
        # Subsample the data to strictly having 10k datapoints
        if balance_data:
            count = 10000/2

            # Find the data that has Q-estimate >= 0.5
            low_data = [x for x in original_data_to_save if x['qestimate'] < 0.5]
            high_data = [x for x in original_data_to_save if x['qestimate'] >= 0.5]
            
            low_data_subsampled = subsample_data(low_data, count, low_or_high="low")
            high_data_subsampled = subsample_data(high_data, count, low_or_high="high")

            print(f"Reducing low_data from {len(low_data)} to {count}")
            print(f"Reducing high_data from {len(high_data)} to {count}")

            input("Press any key to continue...")

            data_to_save = low_data_subsampled + high_data_subsampled

            data_table_10k = pa.Table.from_pylist(data_to_save)
        else:
            data_table_10k = data_table.slice(0, 10000)

        pq.write_table(data_table_10k, os.path.join(outputdir, f"{split_name}_10k.parquet"))


def reduce_and_save_data_for_split_biased(Q_target, outputdir, split_name, balance_data=True, onpolicy_pct_for_success=None):
    """
    Biased in the sense that
        - For low data, we prioritize removing off-policy rollouts (removing the hindsight data that failed)
        - For high data, we prioritize removing on-policy rollouts (removing the agent rollouts that failed)
            We assume that expert's alternative actions would be better
    """
    assert balance_data, "Balance data is required for biased subsampling"

    keys = list(Q_target.keys())
    random.shuffle(keys)

    # This helps us compute the amount that we need to reduce
    original_data_to_save = [
        {'state': Q_target[k]['state'], 
        'reason_action': Q_target[k]['reason_action'], 
        'qestimate': Q_target[k]['qestimate']}
        for k in keys
    ]

    onpolicy_data_to_save = [
        {'state': Q_target[k]['state'], 
        'reason_action': Q_target[k]['reason_action'], 
        'qestimate': Q_target[k]['qestimate']}
        for k in keys if not Q_target[k]['contains_offpolicy']
    ]

    offpolicy_data_to_save = [
        {'state': Q_target[k]['state'], 
        'reason_action': Q_target[k]['reason_action'],
        'qestimate': Q_target[k]['qestimate']}
        for k in keys if Q_target[k]['contains_offpolicy']
    ]

    # Find the data that has Q-estimate >= 0.5
    low_data = [x for x in original_data_to_save if x['qestimate'] < 0.5]
    high_data = [x for x in original_data_to_save if x['qestimate'] >= 0.5]

    # On-policy data
    onpolicy_low_data = [x for x in onpolicy_data_to_save if x['qestimate'] < 0.5]
    onpolicy_high_data = [x for x in onpolicy_data_to_save if x['qestimate'] >= 0.5]

    # Off-policy data
    offpolicy_low_data = [x for x in offpolicy_data_to_save if x['qestimate'] < 0.5]
    offpolicy_high_data = [x for x in offpolicy_data_to_save if x['qestimate'] >= 0.5]

    if split_name == "val":
        count = min(len(low_data), len(high_data))
        low_count_to_reduce = max(0, len(low_data) - count)
        high_count_to_reduce = max(0, len(high_data) - count)
    elif split_name == "train":
        count = 10000/2
        low_count_to_reduce = max(0, len(low_data) - count)
        high_count_to_reduce = max(0, len(high_data) - count)
    else:
        raise ValueError(f"Invalid split name: {split_name}")

    print("-"*50)
    print(f"low_count_to_reduce: {low_count_to_reduce} | high_count_to_reduce: {high_count_to_reduce}")
    print("-"*50)

    # Subsample the low data (off-policy first)
    offpolicy_low_data_subsampled, low_count_left = subsample_data(offpolicy_low_data, low_count_to_reduce, low_or_high="low")
    print(f"offpolicy_low_data_subsampled: {len(offpolicy_low_data_subsampled)} | low_count_left: {low_count_left}")
    onpolicy_low_data_subsampled, _ = subsample_data(onpolicy_low_data, low_count_left, low_or_high="low")
    print(f"onpolicy_low_data_subsampled: {len(onpolicy_low_data_subsampled)}")

    # Combine the data
    low_data_subsampled = offpolicy_low_data_subsampled + onpolicy_low_data_subsampled
    print_histogram([entry['qestimate'] for entry in low_data_subsampled], f"Final Q-estimate for low data (len={len(low_data_subsampled)})", np.linspace(0, 0.5, 6))

    print(f"Reducing low_data from {len(low_data)} to {count} (Reducing {low_count_to_reduce} datapoints in total)\n    (1. offpolicy from {len(offpolicy_low_data)} to {len(offpolicy_low_data_subsampled)} | 2. onpolicy from {len(onpolicy_low_data)} to {len(onpolicy_low_data_subsampled)})\n onpolicy={len(onpolicy_low_data_subsampled)/len(low_data_subsampled):.2f} | offpolicy={len(offpolicy_low_data_subsampled)/len(low_data_subsampled):.2f}")
    input("Press any key to continue...")

    if onpolicy_pct_for_success is not None:
        # Determine the number of on-policy rollouts to include
        onpolicy_count = int(count * onpolicy_pct_for_success)
        offpolicy_count = count - onpolicy_count

        # Determine the amount of data to reduce from the on-policy and off-policy data
        onpolicy_high_count_to_reduce = max(0, len(onpolicy_high_data) - onpolicy_count)
        offpolicy_high_count_to_reduce = max(0, len(offpolicy_high_data) - offpolicy_count)

        # Subsample the on-policy data
        onpolicy_high_data_subsampled, _ = subsample_data(onpolicy_high_data, onpolicy_high_count_to_reduce, low_or_high="high")
        offpolicy_high_data_subsampled, _ = subsample_data(offpolicy_high_data, offpolicy_high_count_to_reduce, low_or_high="high")
    else:
        # Subsample the high data (on-policy first)
        onpolicy_high_data_subsampled, high_count_left = subsample_data(onpolicy_high_data, high_count_to_reduce, low_or_high="high")
        offpolicy_high_data_subsampled, _ = subsample_data(offpolicy_high_data, high_count_left, low_or_high="high")

    # Combine the data
    high_data_subsampled = onpolicy_high_data_subsampled + offpolicy_high_data_subsampled
    print_histogram([entry['qestimate'] for entry in high_data_subsampled], f"Final Q-estimate for high data (len={len(high_data_subsampled)})", np.linspace(0.5, 1, 6))

    print(f"Reducing high_data from {len(high_data)} to {count} (Reducing {high_count_to_reduce} datapoints in total)\n    (1. onpolicy from {len(onpolicy_high_data)} to {len(onpolicy_high_data_subsampled)} | 2. offpolicy from {len(offpolicy_high_data)} to {len(offpolicy_high_data_subsampled)})\n onpolicy={len(onpolicy_high_data_subsampled)/len(high_data_subsampled):.2f} | offpolicy={len(offpolicy_high_data_subsampled)/len(high_data_subsampled):.2f}")
    input("Press any key to continue...")

    data_to_save = low_data_subsampled + high_data_subsampled

    data_table = pa.Table.from_pylist(data_to_save)
    
    # Save the Arrow tables to Parquet files
    os.makedirs(outputdir, exist_ok=True)

    if split_name == "train":
        pq.write_table(data_table, os.path.join(outputdir, f"{split_name}_10k.parquet"))
    else:
        pq.write_table(data_table, os.path.join(outputdir, f"{split_name}.parquet"))


def compute_file_list(rolloutdirs, domain, max_files_per_dir=None, max_rollout_per_task_per_dir_list=None):
    if domain == "alfworld":
        skip_condition = alfworld_skip_file_condition
    elif domain == "twenty_questions":
        skip_condition = twenty_questions_skip_file_condition
    else:
        raise ValueError(f"Invalid domain: {domain}")

    files = []
    for i in range(len(rolloutdirs)):
        rolloutdir = rolloutdirs[i]
        files_per_dir = []
        for file_name in tqdm(os.listdir(rolloutdir)):
            if max_rollout_per_task_per_dir_list is not None:
                max_rollout_per_task_per_dir = max_rollout_per_task_per_dir_list[i]
            else:
                max_rollout_per_task_per_dir = None
            if skip_condition(rolloutdir, file_name, max_rollout_per_task_per_dir):
                continue
            file_path = os.path.join(rolloutdir, file_name)
            files_per_dir.append(file_path)
            if (max_files_per_dir is not None) and (len(files_per_dir) >= max_files_per_dir):
                break
        files = files + files_per_dir

        print(f"Found {len(files_per_dir)} files in {rolloutdir}")
    print(f"== Found {len(files)} files in total ==")

    return files, []

def compute_hindsight_file_list(rolloutdirs, domain, onpolicy_idx_range, offpolicy_idx_range, num_failed_expert_rollouts_to_include=None):
    if domain == "alfworld":
        skip_condition = alfworld_skip_file_condition
        filter_condition_checker = lambda rolloutdir, file_name:False, False
    elif domain == "twenty_questions":
        skip_condition = twenty_questions_skip_file_condition
        filter_condition_checker = twenty_questions_filter_condition_checker
    else:
        raise ValueError(f"Invalid domain: {domain}")

    onpolicy_rollouts_failed = []
    onpolicy_rollouts_succeeded = []
    offpolicy_files_failed = []  # Expert rollout that failed (we could add some rollouts from here to the onpolicy set, so that we have more low-Q datapoints)
    offpolicy_files_good = []

    onpolicy_rollout_idx = []
    for i in range(0, len(onpolicy_idx_range), 2):
        start_idx = onpolicy_idx_range[i]
        end_idx = onpolicy_idx_range[i+1]
        onpolicy_rollout_idx.extend(list(range(start_idx, end_idx)))

    offpolicy_rollout_idx = []
    for i in range(0, len(offpolicy_idx_range), 2):
        start_idx = offpolicy_idx_range[i]
        end_idx = offpolicy_idx_range[i+1]
        offpolicy_rollout_idx.extend(list(range(start_idx, end_idx)))

    print(f"onpolicy_rollout_idx: {onpolicy_rollout_idx}")
    print(f"offpolicy_rollout_idx: {offpolicy_rollout_idx}")

    for i in range(len(rolloutdirs)):
        rolloutdir = rolloutdirs[i]
        for file_name in tqdm(os.listdir(rolloutdir)):
            if skip_condition(rolloutdir, file_name, max_rollout_per_task_per_dir=None):
                continue
            
            failed, expert_action_led_to_success = filter_condition_checker(rolloutdir, file_name)
            file_idx = int(file_name.split("_")[-1].split(".")[0])

            if file_idx in onpolicy_rollout_idx:
                if failed:
                    onpolicy_rollouts_failed.append(os.path.join(rolloutdir, file_name))
                else:
                    onpolicy_rollouts_succeeded.append(os.path.join(rolloutdir, file_name))
            elif file_idx in offpolicy_rollout_idx:
                if expert_action_led_to_success:
                    # Filter out rollouts where the expert directly led to success
                    continue

                if failed:
                    offpolicy_files_failed.append(os.path.join(rolloutdir, file_name))
                else:
                    offpolicy_files_good.append(os.path.join(rolloutdir, file_name))
            else:
                raise ValueError(f"Invalid file index: {file_idx}")
            
    # Shuffle the files
    random.shuffle(onpolicy_rollouts_failed)
    random.shuffle(onpolicy_rollouts_succeeded)
    random.shuffle(offpolicy_files_failed)
    random.shuffle(offpolicy_files_good)

    print(f'========= Raw files collected =========')
    print(f'onpolicy_rollouts_failed: {len(onpolicy_rollouts_failed)} | onpolicy_rollouts_succeeded: {len(onpolicy_rollouts_succeeded)}')
    print(f'offpolicy_files_failed: {len(offpolicy_files_failed)} | offpolicy_files_good: {len(offpolicy_files_good)}')

    # if onpolicy_pct_for_success is not None:
    #     if onpolicy_pct_for_success < 1.0:
    #         # Assume that we are using all of the good offpolicy files
    #         pct_of_offpolicy_to_include = 1 - onpolicy_pct_for_success
    #         total_good_rollouts_needed = int(len(offpolicy_files_good)/pct_of_offpolicy_to_include)
    #         num_onpolicy_to_include = total_good_rollouts_needed - len(offpolicy_files_good)

    #         onpolicy_rollouts_succeeded = onpolicy_rollouts_succeeded[:num_onpolicy_to_include]
    #     else:
    #         onpolicy_rollouts_succeeded = onpolicy_rollouts_succeeded
    #         offpolicy_files_good = []
    
    if num_failed_expert_rollouts_to_include is not None:
        if num_failed_expert_rollouts_to_include != -1:
            offpolicy_files_failed_to_include = offpolicy_files_failed[:num_failed_expert_rollouts_to_include]
        else:
            offpolicy_files_failed_to_include = offpolicy_files_failed
    else:
        offpolicy_files_failed_to_include = []

    files = onpolicy_rollouts_failed + onpolicy_rollouts_succeeded + offpolicy_files_failed_to_include + offpolicy_files_good
    
    print(f"========= Final files collected ==========")
    print(f"onpolicy_rollouts_failed: {len(onpolicy_rollouts_failed)} | onpolicy_rollouts_succeeded: {len(onpolicy_rollouts_succeeded)}")
    print(f"offpolicy_files_failed_to_include: {len(offpolicy_files_failed_to_include)} | offpolicy_files_good: {len(offpolicy_files_good)}")
    print(f"========= Found {len(files)} files in total =========")

    return files, [onpolicy_rollouts_failed, onpolicy_rollouts_succeeded, offpolicy_files_failed_to_include, offpolicy_files_good]
        
@hydra.main(version_base=None, config_path="../../configs", config_name="compute_prm_target.yaml")
def main(cfg: DictConfig):
    random.seed(cfg.seed)

    # Assert only one of train_split or split_name is not None
    assert (cfg.train_split is not None) or (cfg.split_name is not None), "Either train_split or split_name must be provided"
    assert (cfg.train_split is None) or (cfg.split_name is None), "Only one of train_split or split_name can be provided"

    if cfg.split_name is not None:
        # Assume that the rolloutdirs only provide the top directory (that contains 'train', 'val', 'test' subdirectories)
        rolloutdirs = [os.path.join(rolloutdir, cfg.split_name) for rolloutdir in cfg.rolloutdirs]
    else:
        rolloutdirs = cfg.rolloutdirs

    is_hindsight_data = all(["hindsight" in rolloutdir for rolloutdir in rolloutdirs])

    print(f"Confirm the following\n- rolloutdirs: {rolloutdirs}\n- domain: {cfg.domain}\n- outputdir: {cfg.outputdir}\n- is_hindsight_data: {is_hindsight_data}\n{cfg.hindsight if is_hindsight_data else ''}")
    input("Press any key to continue...")

    if is_hindsight_data:
        files, files_breakdown = compute_hindsight_file_list(rolloutdirs, cfg.domain, cfg.hindsight.onpolicy_idx_range, cfg.hindsight.offpolicy_idx_range, cfg.hindsight.num_failed_expert_rollouts_to_include)
    else:
        files, files_breakdown = compute_file_list(rolloutdirs, cfg.domain, cfg.max_files_per_dir, cfg.max_rollout_per_task_per_dir_list)
    
    input("Press any key to continue...")
    compute_prm_target(files, files_breakdown, cfg.domain, cfg.outputdir, cfg.gamma, cfg.cpu_count, cfg.train_split, cfg.split_name, cfg.balance_data, cfg.hindsight.onpolicy_pct_for_success if is_hindsight_data else None, cfg.hindsight.track_offpolicy if is_hindsight_data else False)

if __name__ == "__main__":
    main()
