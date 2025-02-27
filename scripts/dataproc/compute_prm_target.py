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

"""================================================================================
    Alfworld processing functions
================================================================================"""

def alfworld_process_file(file_path, gamma):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        if 'trajectory' not in data:
            return {}

        Q_target = {}
        trajectory = data['trajectory']
        outcome_reward = 2 * trajectory[-1]['score'] - 1  # transform from [-1, 1]
        for t in range(len(trajectory) - 1, -1, -1):
            state, reason_action = alfworld_extract_state_reason_action(trajectory, data['task'], t)
            state_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
            update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, t)
        return Q_target
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}
    
def alfworld_extract_state_reason_action(trajectory, task, t):
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
    reason_action = {
        'reason': trajectory[t]['reason'],
        'action': trajectory[t]['action'],
    }

    return state, reason_action

def alfworld_skip_file_condition(file_name):
    return not file_name.endswith(".json")

"""================================================================================
    20 Questions processing functions
================================================================================"""
def twenty_questions_process_file(file_path, gamma):
    try:
        with open(file_path, 'r') as file:
            trajectory = json.load(file)

        Q_target = {}
        outcome_reward = twenty_questions_normalize_reward(trajectory[-1]['reward'])  # transform from [-1, 1]
        for t in range(len(trajectory) - 1, -1, -1):
            state, reason_action = twenty_questions_extract_state_reason_action(trajectory, t)
            state_hash = sha256(json.dumps({'state': state, 'action': reason_action['action']}, sort_keys=True).encode()).hexdigest()
            update_Q(state, reason_action, state_hash, Q_target, outcome_reward, gamma, k=t, T=len(trajectory))
        return Q_target
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

def twenty_questions_extract_state_reason_action(trajectory, t):
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

    reason_action = {
        'reason': trajectory[t]['reason'],
        'action': trajectory[t]['action'],
    }

    return state, reason_action
    
def twenty_questions_normalize_reward(reward):
    # # normalize reward from [-1, 0] to [-1, 1]
    # return 2 * reward + 1

    # TODO: I think the reward should be in go from [-1, 0] to [0, 1]
    return reward + 1

def twenty_questions_skip_file_condition(file_name):
    return (not file_name.endswith(".json")) or ('_summary_dict' in file_name)


"""================================================================================
    General functions shared by all tasks
================================================================================"""

def print_qestimate_histogram(Q_target, bins=10):
    qestimates = [entry['qestimate'] for entry in Q_target.values()]
    counts, bin_edges = np.histogram(qestimates, bins=bins)
    
    for i in range(len(counts)):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {counts[i]}")

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

# Main function using multiprocessing
def compute_prm_target(files, domain, outputdir, gamma, cpu_count=None, train_split=None, split_name=None):
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

    # # Use multiprocessing Pool for parallel processing
    with Pool(processes=num_cpus) as pool:
        process_func = partial(process_file, gamma=gamma)
        results = list(tqdm(pool.imap(process_func, files), total=len(files), desc="Processing files"))
    
    # First test with single process
    # results = [process_file(file, gamma) for file in files]
    
    # Merge results from all processes
    Q_target = merge_results(results)

    # Transform Q_target back to [0, 1]
    # for key in Q_target.keys():
    #     Q_target[key]['qestimate'] = 0.5 * (Q_target[key]['qestimate'] + 1)

    print_qestimate_histogram(Q_target)

    keys = list(Q_target.keys())
    random.shuffle(keys)

    if split_name is not None:
        data_to_save = [
            {'state': Q_target[k]['state'], 
            'reason_action': Q_target[k]['reason_action'], 
            'qestimate': Q_target[k]['qestimate']}
            for k in keys
        ]

        data_table = pa.Table.from_pylist(data_to_save)
        
        # Save the Arrow tables to Parquet files
        os.makedirs(outputdir, exist_ok=True)
        pq.write_table(data_table, os.path.join(outputdir, f"{split_name}.parquet"))
        
        print(f"Saving {len(data_to_save)} datapoints as {split_name}.parquet")
        
        if split_name == "train":
            # Subsample the data to strictly having 10k datapoints
            pq.write_table(data_table.slice(0, 10000), os.path.join(outputdir, f"{split_name}_10k.parquet"))
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
    compute_prm_target(files, cfg.domain, cfg.outputdir, cfg.gamma, cfg.cpu_count, cfg.train_split, cfg.split_name)

if __name__ == "__main__":
    main()
