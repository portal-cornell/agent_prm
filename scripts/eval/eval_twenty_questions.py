"""
Typical usage:

# Online eval
python scripts/eval/eval_twenty_questions.py mode=online host_sglang=true data_types=[val,test] online.rollout_per_task=1 online.num_alt_responses=5 online.batch_size=32 elogger=true

# Consolidate online eval
python scripts/eval/eval_twenty_questions.py mode=consolidate_online consolidate_online.table_notes='till-pi1'
"""

import os
import time
import math
import shutil
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
from typing import List, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import signal
from agent_prm.agents.agent_registry import initialize_agent
from agent_prm.agents.agent import Agent
from agent_prm.utils.parser import parse_reason_and_action_twenty_questions
from agent_prm.utils.cfg_utils import get_output_folder_name
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT, TEST_OBJECT_DICT, WordVariants, get_default_word_list
from agent_prm.envs.twenty_questions.env import setup_twenty_questions_env, setup_batched_twenty_questions_env
from agent_prm.envs.twenty_questions.interface import rollout_batch
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import start_sglang_server
from agent_prm.utils.cfg_utils import get_output_path, find_matching_iter

def offline_eval(cfg: dict, agent: Agent):
    """
    Load the dataset and evaluate the model
    """
    pass


def setup_sglang_server(agent_config: dict):
    """
    Setup the SGLang server

    If the agent is a SGLang server, we only need to start one server
    If the agent is a Best of N, we need to start two servers
        - One for the generator
        - One for the critic

    Returns:
        a list of processes
    """
    processes = []
    if agent_config.type == "sglang_server":
        if "TODO" in agent_config.server_url:
            port = None
        else:
            port = int(agent_config.server_url.split(":")[-1][:-1])

        print(f"Starting SGLang server on port {port}")
        process, server_url, _ = start_sglang_server(model_path=agent_config.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.dist_url_port)
        
        if "TODO" in agent_config.server_url:
            agent_config.server_url = server_url

        processes.append(process)
    elif agent_config.type == "best_of_n" or agent_config.type == "sglang_server_with_critic":
        # Start the critic
        if "TODO" in agent_config.critic.server_url:
            port = None
        else:
            port = int(agent_config.critic.server_url.split(":")[-1][:-1])

        print(f"Starting SGLang server for the critic on port {port}, serving on the highest ID GPU")
        process, server_url, base_gpu_id = start_sglang_server(model_path=agent_config.critic.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.critic.dist_url_port)
        
        if "TODO" in agent_config.critic.server_url:
            agent_config.critic.server_url = server_url

        processes.append(process)

        # Start the generator
        #    One condition to not host sglang: if generator has the field host_sglang and it is False
        gpu_id = max(0, base_gpu_id - 1)
        if agent_config.type == "best_of_n":
            if not (hasattr(agent_config.generator, 'host_sglang') and not agent_config.generator.host_sglang):
                if "TODO" in agent_config.generator.server_url:
                    port = None
                else:
                    port = int(agent_config.generator.server_url.split(":")[-1][:-1])

                print(f"Starting SGLang server for the generator on port {port}, serving on the next highest ID GPU {gpu_id}")
                process, server_url, _ = start_sglang_server(model_path=agent_config.generator.model_id,
                                                        port=port, 
                                                        tp=1,
                                                        dist_url_port=agent_config.generator.dist_url_port,
                                                        gpu_id=gpu_id)
                
                if "TODO" in agent_config.generator.server_url:
                    agent_config.generator.server_url = server_url
                
                processes.append(process)
        elif agent_config.type == "sglang_server_with_critic":
            if "TODO" in agent_config.server_url:
                port = None
            else:
                port = int(agent_config.server_url.split(":")[-1][:-1])

            print(f"Starting SGLang server for the critic on port {port}, serving on the highest ID GPU")
            process, server_url, _ = start_sglang_server(model_path=agent_config.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.dist_url_port,
                                                gpu_id=base_gpu_id)
            
            if "TODO" in agent_config.server_url:
                agent_config.server_url = server_url

            processes.append(process)

    return processes


def online_eval(cfg: dict, logdir: str, agent: Agent):
    """
    Evaluate the model by interacting with the environment
    """
    batched_env = setup_batched_twenty_questions_env(host=cfg.sim_host, port=cfg.sim_port)
    all_obj_list = [wv[0] for wv in get_default_word_list("all")]
    bs = cfg.online.batch_size

    # Collect the entire list of all the objects that we are evaluating on. This helps more efficiently use the batch size
    #   Each element is a tuple of (object, data_type, rollout_idx)
    all_objects_to_eval_on = []
    for data_type in cfg.data_types:
        os.makedirs(os.path.join(logdir, data_type), exist_ok=True)

        summary_dict_fp = os.path.join(logdir, data_type, "_summary_dict.json")
        if not os.path.exists(summary_dict_fp):
            print(f"Summary dict not found at {summary_dict_fp}. Creating a new one.")
            summary_dict = {}
            save_json(summary_dict_fp, summary_dict)
        else:
            print(f"Loading summary dict from {summary_dict_fp}")
            summary_dict = load_json(summary_dict_fp)

        if data_type == "train":
            object_dict_to_use = TRAIN_OBJECT_DICT
        elif data_type == "val":
            object_dict_to_use = VALIDATION_OBJECT_DICT
        elif data_type == "test":
            object_dict_to_use = TEST_OBJECT_DICT
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
        for rollout_idx in range(cfg.online.rollout_per_task_range_min, cfg.online.rollout_per_task_range_max):
            rollout_idx_str = str(rollout_idx)
            if rollout_idx_str not in summary_dict:
                summary_dict[rollout_idx_str] = []

            # Consolidate the objects to evaluate on
            objects_to_eval_on = [(obj, data_type, rollout_idx, category) for category in object_dict_to_use.keys() for obj in object_dict_to_use[category] if obj not in summary_dict[rollout_idx_str]]
            all_objects_to_eval_on.extend(objects_to_eval_on)

    for batch in tqdm(range(math.ceil(len(all_objects_to_eval_on) / bs))):
        # Determine the objects to evaluate on for this batch
        batch_objects_tuples = all_objects_to_eval_on[batch * bs:(batch + 1) * bs]
        batch_objects = [obj for obj, _, _, _ in batch_objects_tuples]  # Used to initialize the environment
        batch_obj_categories = [category for _, _, _, category in batch_objects_tuples]

        print(f"=========== Batch {batch} has {len(batch_objects_tuples)} objects: {batch_objects_tuples} ===========")
        
        histories, words_to_guess = batched_env.reset(num_envs=len(batch_objects), words_to_guess=[WordVariants.from_str(obj) for obj in batch_objects])

        # Initialize prev_dones as a list of False with the same length as batch_objects
        prev_dones = [False for _ in range(len(batch_objects))]
        traj_list = [[] for _ in range(len(batch_objects))]

        traj_list = rollout_batch(agent, batched_env, all_obj_list, 
                                  words_to_guess, batch_obj_categories, histories, 
                                  traj_list, prev_dones, cfg.online.num_alt_responses)

        # Save the trajectories
        for i in range(len(batch_objects)):
            obj, obj_data_type, obj_rollout_idx, _ = batch_objects_tuples[i]
            obj_rollout_idx_str = str(obj_rollout_idx)

            # Open up the correct summary dict
            summary_dict_fp = os.path.join(logdir, obj_data_type, "_summary_dict.json")
            summary_dict = load_json(summary_dict_fp)

            # Update the summary dict
            if obj_rollout_idx_str not in summary_dict:
                summary_dict[obj_rollout_idx_str] = []

            summary_dict[obj_rollout_idx_str].append(obj)
            save_json(summary_dict_fp, summary_dict)
            
            # Save the trajectory
            save_json(os.path.join(logdir, obj_data_type, f"{obj}_{obj_rollout_idx_str}.json"), traj_list[i])


def consolidate_online_eval(cfg: dict, table_fp: str, agent_rollout_dir: str, agent_name: str, rollout_per_task_dict: Dict[str, int], use_existing_table: bool = False, overwrite_existing_entry: bool = False):
    """
    Consolidate the online eval results and save it as a csv file
    """
    if not os.path.exists(table_fp):
        table_dict = {
            "model": [],
            "train (avg reward)": [],
            "train (se reward)": [],
            "train (avg success rate)": [],
            "train (se success rate)": [],
            "val (avg reward)": [],
            "val (se reward)": [],
            "val (avg success rate)": [],
            "val (se success rate)": [],
            "test (avg reward)": [],
            "test (se reward)": [],
            "test (avg success rate)": [],
            "test (se success rate)": [],
            "total (avg reward)": [],
            "total (se reward)": [],
            "total (avg success rate)": [],
            "total (se success rate)": [],
            "train (enough rollouts)": [],
            "val (enough rollouts)": [],
            "test (enough rollouts)": []
        }
    else:
        table = pd.read_csv(table_fp)
        table_dict = table.to_dict(orient="list")
    
    # Check if the agent_name is already in the table
    if use_existing_table and (agent_name in table_dict["model"]) and not overwrite_existing_entry:
        print(f"Agent {agent_name} already exists in the table. Skipping the consolidation.")
    else:
        if agent_name in table_dict["model"]:
            print(f"Agent {agent_name} already exists in the table. Overwriting the existing row.")
            overwrite = True
        else:
            table_dict["model"].append(agent_name)
            overwrite = False

        def is_valid_rollout(f: str, data_type: str) -> bool:
            """
            Check if the rollout is valid
            """
            is_a_rollout_file = f.endswith(".json") and not f.endswith("_summary_dict.json")
            to_include = False

            for i in range(rollout_per_task_dict[data_type]):
                if f"_{i}" in f:
                    to_include = True
                    break

            return is_a_rollout_file and to_include

        total_rewards = []
        total_success_rates = []

        for data_type in ["train", "val", "test"]:
            all_rewards = []

            if os.path.exists(os.path.join(agent_rollout_dir, data_type)):
                # Get all the rollouts that are used to consolidate the results
                json_files = [f for f in os.listdir(os.path.join(agent_rollout_dir, data_type)) if is_valid_rollout(f, data_type)]

                if not any([f.endswith(f"{rollout_per_task_dict[data_type]-1}.json") for f in json_files]):
                    print(f"WARNING: {agent_name} does not have all the rollouts for {data_type}, which needs {rollout_per_task_dict[data_type]} rollouts per task")
                    has_enough_rollouts = False
                else:
                    has_enough_rollouts = True

                # Compute rewards efficiently
                all_rewards = [sum(t["reward"] for t in load_json(os.path.join(agent_rollout_dir, data_type, f))) for f in json_files]
                mean_reward = np.mean(all_rewards)
                se_reward = np.std(all_rewards)/math.sqrt(len(all_rewards))

                # Compute success rate
                all_success_rates = [load_json(os.path.join(agent_rollout_dir, data_type, f))[-1]["reward"] == 0 for f in json_files]
                mean_success_rate = np.mean(all_success_rates)
                se_success_rate = np.std(all_success_rates)/math.sqrt(len(all_success_rates))
            else:
                print(f"WARNING: {agent_name} does not have any rollouts for {data_type}")
                mean_reward = np.nan
                se_reward = np.nan
                mean_success_rate = np.nan
                se_success_rate = np.nan
                has_enough_rollouts = False

            if overwrite:
                table_dict[f"{data_type} (avg reward)"][table_dict["model"].index(agent_name)] = mean_reward
                table_dict[f"{data_type} (se reward)"][table_dict["model"].index(agent_name)] = se_reward
                table_dict[f"{data_type} (avg success rate)"][table_dict["model"].index(agent_name)] = mean_success_rate
                table_dict[f"{data_type} (se success rate)"][table_dict["model"].index(agent_name)] = se_success_rate
                table_dict[f"{data_type} (enough rollouts)"][table_dict["model"].index(agent_name)] = has_enough_rollouts
            else:
                table_dict[f"{data_type} (avg reward)"].append(mean_reward)
                table_dict[f"{data_type} (se reward)"].append(se_reward)
                table_dict[f"{data_type} (avg success rate)"].append(mean_success_rate)
                table_dict[f"{data_type} (se success rate)"].append(se_success_rate)
                table_dict[f"{data_type} (enough rollouts)"].append(has_enough_rollouts)
            
            total_rewards.append(mean_reward)
            total_success_rates.append(mean_success_rate)

        total_avg_reward = np.mean(total_rewards)
        total_se_reward = np.std(total_rewards)/math.sqrt(len(total_rewards))
        total_avg_success_rate = np.mean(total_success_rates)
        total_se_success_rate = np.std(total_success_rates)/math.sqrt(len(total_success_rates))

        if overwrite:
            table_dict["total (avg reward)"][table_dict["model"].index(agent_name)] = total_avg_reward
            table_dict["total (se reward)"][table_dict["model"].index(agent_name)] = total_se_reward
            table_dict["total (avg success rate)"][table_dict["model"].index(agent_name)] = total_avg_success_rate
            table_dict["total (se success rate)"][table_dict["model"].index(agent_name)] = total_se_success_rate
        else:
            table_dict["total (avg reward)"].append(total_avg_reward)
            table_dict["total (se reward)"].append(total_se_reward)
            table_dict["total (avg success rate)"].append(total_avg_success_rate)
            table_dict["total (se success rate)"].append(total_se_success_rate)

        # Convert the table dict to a dataframe and save it
        table = pd.DataFrame(table_dict)
        table.to_csv(table_fp, index=False)


@hydra.main(version_base=None, config_path="../../configs/eval_config", config_name="twenty_questions.yaml")
def main(cfg: DictConfig):
    elogger.set_activate(cfg.elogger)

    if cfg.mode == "consolidate_online":
        # The table for this hydra run is saved in the hydra folder
        hydra_folder_path = get_output_path()
        table_fp = os.path.join(hydra_folder_path, f"online_eval_table{'_' + cfg.consolidate_online.table_notes if cfg.consolidate_online.table_notes else ''}.csv")

        if cfg.consolidate_online.use_existing_table:
            # Save a copy of the existing table in the current folder
            table = pd.read_csv(os.path.join(cfg.logdir, f"online_eval_table{'_' + cfg.consolidate_online.main_table_notes if cfg.consolidate_online.main_table_notes else ''}.csv"))

            # Save a copy of the existing table in the current folder
            table.to_csv(table_fp, index=False)

    # Load the model
    print(f"Mode={cfg.mode}, for agents: {[agent_config.log_name for agent_config in cfg.agents]}")
    for agent_i in tqdm(range(len(cfg.agents))):
        agent_config = cfg.agents[agent_i]
        if cfg.host_sglang:
            processes = setup_sglang_server(agent_config)
        # Extract the agent_name and logdir
        if agent_config.type == "gpt4o_expert":
            # Use the data collected for SFT
            assert cfg.mode == "consolidate_online", "Gpt4o expert can only be used in consolidate_online mode, where we are comparing the performance of different models"
            logdir = os.path.join(cfg.rollout_data_dir, f"iter{cfg.iter}")
            agent_name = "gpt4o_expert"
        else:
            agent_name = agent_config.model_id if agent_config.type != "best_of_n" else agent_config.generator.model_id
            
            if "checkpoint" in agent_name:
                agent_name = os.path.basename(agent_name.split("/")[-2])
            else:
                agent_name = os.path.basename(agent_name)

            agent_name = f"{agent_config.log_name}_{agent_name}"

            logdir = os.path.join(cfg.logdir, find_matching_iter(agent_config.log_name), agent_name)
        
        os.makedirs(logdir, exist_ok=True)

        if cfg.mode == "consolidate_online":
            consolidate_online_eval(cfg, table_fp, agent_rollout_dir=logdir, agent_name=agent_config.log_name, rollout_per_task_dict=cfg.consolidate_online.rollout_per_task_dict, use_existing_table=cfg.consolidate_online.use_existing_table, overwrite_existing_entry=cfg.consolidate_online.overwrite_existing_entry)

            # Check if the file or symlink exists, then remove it
            dst_link_fp = os.path.join(cfg.logdir, f"online_eval_table{'_' + cfg.consolidate_online.main_table_notes if cfg.consolidate_online.main_table_notes else ''}.csv")
            if os.path.exists(dst_link_fp) or os.path.islink(dst_link_fp):
                os.remove(dst_link_fp)

            # Copy the table to the dst_link_fp
            shutil.copy(table_fp, dst_link_fp)
        else:
            try:
                agent = initialize_agent(agent_config,
                                            parse_reason_action_fn=parse_reason_and_action_twenty_questions,
                                            verbose=cfg["verbose"],
                                            debug=cfg["debug"])
                
                print(f"Evaluating {agent_name} in {logdir}")

                if cfg.mode == "offline":
                    offline_eval(cfg, logdir, agent)
                elif cfg.mode == "online":
                    online_eval(cfg, logdir, agent)
                else:
                    raise ValueError(f"Invalid mode: {cfg.mode}")
            except Exception as e:
                elogger.log(f"Error: {e}")
                raise e
            
        if cfg.host_sglang:
            if processes is not None:
                for process in processes:
                    # Cleanup the SGLang server
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait()
                    print("SGLang server terminated")

    if cfg.mode == "online":
        # Because this takes a long time, we notify when the online eval is done
        elogger.log(f"Online eval results saved for Agents: {[agent_config.log_name for agent_config in cfg.agents]}")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        elogger.log(f"Error: {e}")
        raise e
