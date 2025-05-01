"""
Typical usage:

# Online eval
python scripts/eval/eval_car_dealer.py mode=online host_sglang=true data_types=[train] online.rollout_per_task_range_min=0 online.rollout_per_task_range_max=1 online.num_alt_responses=0 online.batch_size=32 elogger=true sim_host=TODO sim_port=TODO local_sglang=true

# Consolidate online eval
python scripts/eval/eval_car_dealer.py mode=consolidate_online consolidate_online.use_existing_table=true consolidate_online.overwrite_existing_entry=true consolidate_online.table_notes='TODO' consolidate_online.main_table_notes=''
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
from jinja2 import Template

from agent_prm.agents.agent_registry import initialize_agent
from agent_prm.agents.agent import Agent
from agent_prm.utils.parser import parse_reason_and_action_twenty_questions
from agent_prm.utils.cfg_utils import get_output_folder_name
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import setup_sglang_server 
from agent_prm.utils.cfg_utils import get_output_path, find_matching_iter

from agent_prm.envs.car_dealer.env import setup_batched_car_dealer_env
from agent_prm.envs.car_dealer.data import get_all_games_to_play, load_car_inventories
from agent_prm.envs.car_dealer.interface import rollout_batch


def online_eval(cfg: dict, logdir: str, agent: Agent, agent_api_call_template: Template, agent_prompt_template: Template):
    """
    Evaluate the model by interacting with the environment
    """
    batched_env = setup_batched_car_dealer_env(host=cfg.sim_host, port=cfg.sim_port)
    bs = cfg.online.batch_size

    car_inventories = load_car_inventories()

    # A list of tuples
    #  (rollout_idx, game_id, data_type, buyer_info)
    all_games_to_play_list = get_all_games_to_play(cfg.data_types, logdir, range(cfg.online.rollout_per_task_range_min, cfg.online.rollout_per_task_range_max))

    for batch in tqdm(range(math.ceil(len(all_games_to_play_list) / bs)), desc="Batches"):
        # Determine the games to play for this batch
        # Tuple: (rollout_idx, game_id, data_type, buyer_info)
        batch_games_to_play_list = all_games_to_play_list[batch * bs:(batch + 1) * bs]
        batch_buyer_infos = [buyer_info for _, _, _, buyer_info in batch_games_to_play_list]  # Used to initialize the environment
        
        print(f"=========== Batch {batch} has {len(batch_games_to_play_list)} objects: {[(rollout_idx, game_id, data_type) for rollout_idx, game_id, data_type, _ in batch_games_to_play_list]} ===========")
        
        histories = batched_env.reset(buyer_infos=batch_buyer_infos)

        # Initialize prev_dones as a list of False with the same length as batch_objects
        prev_dones = [False for _ in range(len(batch_buyer_infos))]
        traj_list = [[] for _ in range(len(batch_buyer_infos))]

        traj_list = rollout_batch(agent_api_call_template, agent_prompt_template, 
                                  agent, batched_env, car_inventories, batch_buyer_infos, 
                                  histories, traj_list, prev_dones, cfg.online.num_alt_responses)

        # Save the trajectories
        for i in range(len(batch_games_to_play_list)):
            rollout_idx, game_id, data_type, _, = batch_games_to_play_list[i]
            rollout_idx_str = str(rollout_idx)

            # Open up the correct summary dict
            summary_dict_fp = os.path.join(logdir, data_type, "_summary_dict.json")
            summary_dict = load_json(summary_dict_fp)

            # Update the summary dict
            if rollout_idx_str not in summary_dict:
                summary_dict[rollout_idx_str] = []

            summary_dict[rollout_idx_str].append(game_id)
            save_json(summary_dict_fp, summary_dict)
            
            # Save the trajectory
            save_json(os.path.join(logdir, data_type, f"{game_id}_{rollout_idx_str}.json"), traj_list[i])



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
            is_a_rollout_file = f.endswith(".json") and not f.endswith("_summary_dict.json") and not "original" in f
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
                all_success_rates = [load_json(os.path.join(agent_rollout_dir, data_type, f))[-1]["success"] for f in json_files]
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


@hydra.main(version_base=None, config_path="../../configs/eval_config", config_name="car_dealer.yaml")
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
    if cfg.mode == "consolidate_online":
        agent_iter = tqdm(range(len(cfg.agents)), position=0, desc="Agents")
    else:
        agent_iter = range(len(cfg.agents))

    for agent_i in agent_iter:
        agent_config = cfg.agents[agent_i]
        if cfg.host_sglang:
            processes = setup_sglang_server(agent_config, cfg.local_sglang)
        # Extract the agent_name and logdir
        if agent_config.type == "gpt4o_expert":
            # Use the data collected for SFT
            assert cfg.mode == "consolidate_online", "Gpt4o expert can only be used in consolidate_online mode, where we are comparing the performance of different models"
            logdir = os.path.join(cfg.logdir, "baseline", agent_config.log_name)
            agent_name = "gpt4o_expert"
        else:
            if agent_config.type == "best_of_n":
                agent_name = agent_config.generator.model_id
            elif agent_config.type == "dual_sglang_server_agents":
                agent_name = agent_config.response_generator.model_id
            else:
                agent_name = agent_config.model_id
            
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
                                            parse_reason_action_fn=lambda x: x, # Placeholder. This is getting set in rollout_batch (because we both need to call the API and also generate responses to the user)
                                            verbose=cfg["verbose"],
                                            debug=cfg["debug"])
                
                print(f"Evaluating {agent_name} in {logdir}")

                if cfg.mode == "online":
                    with open(agent_config.api_prompt_template_file, "r") as f:
                        agent_api_call_template = Template(f.read())

                    with open(agent_config.prompt_template_file, "r") as f:
                        agent_prompt_template = Template(f.read())

                    online_eval(cfg, logdir, agent, agent_api_call_template, agent_prompt_template)
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
