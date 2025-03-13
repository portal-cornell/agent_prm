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

"""================================================================================
    Alfworld processing functions
================================================================================"""

def alfworld_process_file(file_path, gamma, exclude_reason=False):
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

def alfworld_skip_file_condition(file_name):
    return not file_name.endswith(".json")

def alfworld_success_file_condition(file_name):
    # TODO: Implement this
    return True

"""================================================================================
    20 Questions processing functions
================================================================================"""
def twenty_questions_process_file(file_path, gamma, exclude_reason=False):
    try:
        with open(file_path, 'r') as file:
            trajectory = json.load(file)

        Q_target = {}
        outcome_reward = twenty_questions_normalize_reward(trajectory[-1]['reward'])  # transform from [-1, 1]
        for t in range(len(trajectory) - 1, -1, -1):
            state, reason_action = twenty_questions_extract_state_reason_action(trajectory, t, exclude_reason=exclude_reason)
            state_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
            update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, k=t, T=len(trajectory))
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

def twenty_questions_skip_file_condition(file_name):
    return (not file_name.endswith(".json")) or ('_summary_dict' in file_name)

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

def update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, k, T):
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
            'count': 0
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
    return merged_Q_target

def subsample_data(data: List[Dict], count: int, bins: int = 5, low_or_high: str = "low"):
    """
    Parameters:
        data: List of dictionaries, each containing 'qestimate' and 'state' and 'reason_action'
        count: The number of datapoints to subsample to
        bins: The number of bins to use for the histogram
    """
    if len(data) > count:
        q_val_list = [entry['qestimate'] for entry in data]

        num_datapoints_to_reduce = len(q_val_list) - count
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

        while total_removal_count < num_datapoints_to_reduce:
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
            
            count = 0
            new_data = []
            for entry in data_to_process:
                if entry['qestimate'] >= bin_to_remove_min and entry['qestimate'] < bin_to_remove_max and count < removal_count:
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

        return new_data
    else:
        return data

    
# Main function using multiprocessing
def compute_prm_target(files, domain, outputdir, gamma, cpu_count=None, train_split=None, split_name=None, balance_data=False):
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
    print_count_histogram(Q_target, bins=np.array(list(range(1, 6)) + list(range(6, 10, 2)) +list(range(10, 100, 10)) + list(range(100, max([x['count'] for x in Q_target.values()]), 100))))

    keys = list(Q_target.keys())
    random.shuffle(keys)

    if split_name is not None:
        original_data_to_save = [
            {'state': Q_target[k]['state'], 
            'reason_action': Q_target[k]['reason_action'], 
            'qestimate': Q_target[k]['qestimate']}
            for k in keys
        ]

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
        
        if split_name == "train":
            # Subsample the data to strictly having 10k datapoints
            if balance_data:
                count = 10000/2
                
                low_data_subsampled = subsample_data(low_data, count, low_or_high="low")
                high_data_subsampled = subsample_data(high_data, count, low_or_high="high")

                print(f"Reducing low_data from {len(low_data)} to {count}")
                print(f"Reducing high_data from {len(high_data)} to {count}")

                data_to_save = low_data_subsampled + high_data_subsampled

                data_table_10k = pa.Table.from_pylist(data_to_save)
            else:
                data_table_10k = data_table.slice(0, 10000)

            pq.write_table(data_table_10k, os.path.join(outputdir, f"{split_name}_10k.parquet"))
    else:
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

def compute_file_list(rolloutdirs, domain, max_files_per_dir=None):
    if domain == "alfworld":
        skip_condition = alfworld_skip_file_condition
    elif domain == "twenty_questions":
        skip_condition = twenty_questions_skip_file_condition
    else:
        raise ValueError(f"Invalid domain: {domain}")

    files = []
    for rolloutdir in rolloutdirs:
        files_per_dir = []
        for file_name in tqdm(os.listdir(rolloutdir)):
            if skip_condition(file_name):
                continue
            file_path = os.path.join(rolloutdir, file_name)
            files_per_dir.append(file_path)
            if (max_files_per_dir is not None) and (len(files_per_dir) >= max_files_per_dir):
                break
        files = files + files_per_dir

    return files
        
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

    files = compute_file_list(rolloutdirs, cfg.domain, cfg.max_files_per_dir)
    compute_prm_target(files, cfg.domain, cfg.outputdir, cfg.gamma, cfg.cpu_count, cfg.train_split, cfg.split_name, cfg.balance_data)

if __name__ == "__main__":
    main()
