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
from agent_prm.utils.logger_email import elogger
from agent_prm.utils.general_utils import start_sglang_server
from agent_prm.utils.cfg_utils import get_output_path

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
        port = int(agent_config.server_url.split(":")[-1][:-1])
        print(f"Starting SGLang server on port {port}")
        process, _, _ = start_sglang_server(model_path=agent_config.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.dist_url_port)
        processes.append(process)
    elif agent_config.type == "best_of_n":
        # Start the generator
        port = int(agent_config.generator.server_url.split(":")[-1][:-1])
        print(f"Starting SGLang server for the generator on port {port}, serving on the highest ID GPU")
        process, _, base_gpu_id = start_sglang_server(model_path=agent_config.generator.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.generator.dist_url_port)
        processes.append(process)

        # Start the critic
        port = int(agent_config.critic.server_url.split(":")[-1][:-1])
        gpu_id = max(0, base_gpu_id - 1)
        print(f"Starting SGLang server for the critic on port {port}, serving on the next highest ID GPU {gpu_id}")
        process, _, _ = start_sglang_server(model_path=agent_config.critic.model_id,
                                                port=port, 
                                                tp=1,
                                                dist_url_port=agent_config.critic.dist_url_port,
                                                gpu_id=gpu_id)
        processes.append(process)

    return processes

        
def query_agent(agent: Agent, history: List[Dict[str, str]], all_obj_list: List[WordVariants], last_question: bool = False):
    """
    Query the agent for a reason and action
    """
    input_data = {
        'mode': 'input' if not last_question else 'input_final',
        'all_obj_list': all_obj_list,
        'observation_action_history': history
    }

    reason, action = agent.predict_reason_action(input_data)

    return reason, action


def query_agent_batch(agent: Agent, histories: List[List[Dict[str, str]]], all_obj_list: List[WordVariants], last_question: bool = List[bool], num_alt_responses: int = 0):
    """
    Query the agent in batch
    """
    input_datas = [
        {
            'mode': 'input' if not last_question[i] else 'input_final',
            'all_obj_list': all_obj_list,
            'observation_action_history': histories[i]
        }
        for i in range(len(histories))
    ]

    reason_actions, generated_texts = agent.predict_reason_action_batch(input_datas, num_responses=1 + num_alt_responses, alt_temperature_for_extra_responses=1.0 if num_alt_responses > 0 else None)

    return reason_actions, generated_texts

def online_eval(cfg: dict, logdir: str, agent: Agent):
    """
    Evaluate the model by interacting with the environment
    """
    rollout_per_obj = cfg.online.rollout_per_task
    batched_env = setup_batched_twenty_questions_env(port=cfg.sim_port)
    all_obj_list = [wv[0] for wv in get_default_word_list("all")]
    bs = cfg.online.batch_size

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
        
        for rollout_idx in range(rollout_per_obj):
            rollout_idx_str = str(rollout_idx)
            if rollout_idx_str not in summary_dict:
                summary_dict[rollout_idx_str] = []

            # Consolidate the objects to evaluate on
            objects_to_eval_on = [obj for category in object_dict_to_use.keys() for obj in object_dict_to_use[category] if obj not in summary_dict[rollout_idx_str]]

            for batch in tqdm(range(math.ceil(len(objects_to_eval_on) / bs))):
                # Determine the objects to evaluate on for this batch
                batch_objects = objects_to_eval_on[batch * bs:(batch + 1) * bs]

                print(f"=========== Batch {batch} has {len(batch_objects)} objects: {batch_objects} ===========")
                
                histories, words_to_guess = batched_env.reset(num_envs=len(batch_objects), words_to_guess=[WordVariants.from_str(obj) for obj in batch_objects])

                # Initialize prev_dones as a list of False with the same length as batch_objects
                prev_dones = [False for _ in range(len(batch_objects))]
                traj_list = [[] for _ in range(len(batch_objects))]

                while not all(prev_dones):
                    # Batched way
                    start_time = time.time()
                    reasons, actions, raw_texts, scores = [], [], [], []
                    if cfg.online.num_alt_responses > 0:
                        alt_reasons, alt_actions, alt_raw_texts, alt_scores = [[] for _ in range(len(batch_objects))], [[] for _ in range(len(batch_objects))], [[] for _ in range(len(batch_objects))], [[] for _ in range(len(batch_objects))]  # For each object, we have a list of alt_reasons and alt_actions and alt_scores
                    last_questions = [len(histories[i]) == batched_env.max_conversation_length - 1 for i in range(len(batch_objects))]
                    reasons_actions_dict, generated_raw_texts = query_agent_batch(agent, histories, all_obj_list, last_questions, cfg.online.num_alt_responses)

                    # print(f"generated_raw_texts: {generated_raw_texts}")
                    # print(f"len(generated_raw_texts): {len(generated_raw_texts)}")
                    # input("stop")
                    for i in range(len(batch_objects)):
                        if prev_dones[i]:
                            reasons.append("")
                            actions.append("")
                            scores.append(None)
                            raw_texts.append("")
                        else:
                            reasons.append(reasons_actions_dict[i][0]["reason"])
                            actions.append(reasons_actions_dict[i][0]["action"])
                            raw_texts.append(generated_raw_texts[i*(1+cfg.online.num_alt_responses)]) 
                            has_critic_score = "score" in reasons_actions_dict[i][0]

                            # print(f"raw_text @({i*(1+cfg.online.num_alt_responses)}): {generated_raw_texts[i*(1+cfg.online.num_alt_responses)]}")
                            # print(f"reason: {reasons[-1]}")
                            # print(f"action: {actions[-1]}")
                            # input("stop")

                            if has_critic_score:
                                scores.append(reasons_actions_dict[i][0]["score"])
                            else:
                                scores.append(None)
                            
                            if cfg.online.num_alt_responses > 0:
                                for j in range(cfg.online.num_alt_responses):
                                    alt_reasons[i].append(reasons_actions_dict[i][j+1]["reason"])
                                    alt_actions[i].append(reasons_actions_dict[i][j+1]["action"])
                                    alt_raw_texts[i].append(generated_raw_texts[i*(1+cfg.online.num_alt_responses) + j + 1])

                                    # print(f"alt_raw_text @({i*(1+cfg.online.num_alt_responses) + 1 + j}): {generated_raw_texts[i*(1+cfg.online.num_alt_responses) + j + 1]}")
                                    # print(f"alt_reason: {alt_reasons[i][-1]}")
                                    # print(f"alt_action: {alt_actions[i][-1]}")
                                    # input("alt")
                                    
                                    if has_critic_score:
                                        alt_scores[i].append(reasons_actions_dict[i][j+1]["score"])
                                    else:
                                        alt_scores[i].append(None)

                    print(f"[AGENT] time taken for batch_size={bs}: {time.time() - start_time}")

                    # Step the environment
                    histories, answer_reasons, answers, rewards, dones = batched_env.step(words_to_guess, histories, actions, prev_dones)

                    # Log the trajectories
                    for i in range(len(batch_objects)):
                        if not prev_dones[i]:
                            traj_list[i].append({
                                "step": len(traj_list[i]),
                                "reason": reasons[i],
                                "action": actions[i],
                                "raw_text": raw_texts[i],
                                "answerer_reason": answer_reasons[i],
                                "answer": answers[i],
                                "reward": rewards[i],
                                "score": scores[i],
                                'alternatives': [
                                    {
                                        "reason": alt_reasons[i][j],
                                        "action": alt_actions[i][j],
                                        "raw_text": alt_raw_texts[i][j],
                                        "score": alt_scores[i][j]
                                    }
                                    for j in range(cfg.online.num_alt_responses)
                                ] if cfg.online.num_alt_responses > 0 else None
                            })

                    prev_dones = dones

                # Update the summary dict
                summary_dict[rollout_idx_str].extend(batch_objects)
                save_json(summary_dict_fp, summary_dict)

                # Save the trajectories
                for i in range(len(batch_objects)):
                    save_json(os.path.join(logdir, data_type, f"{batch_objects[i]}_{rollout_idx_str}.json"), traj_list[i])

def consolidate_online_eval(cfg: dict, table_fp: str, agent_rollout_dir: str, agent_name: str, rollout_per_task: int = 3):
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
            "total (se success rate)": []
        }
    else:
        table = pd.read_csv(table_fp)
        table_dict = table.to_dict(orient="list")
    
    # Check if the agent_name is already in the table
    if agent_name in table_dict["model"]:
        print(f"Agent {agent_name} already exists in the table. Overwriting the existing row.")
        overwrite = True
    else:
        table_dict["model"].append(agent_name)
        overwrite = False

    def is_valid_rollout(f: str) -> bool:
        """
        Check if the rollout is valid
        """
        is_a_rollout_file = f.endswith(".json") and not f.endswith("_summary_dict.json")
        to_include = False

        for i in range(rollout_per_task):
            if f"_{i}" in f:
                to_include = True
                break

        return is_a_rollout_file and to_include

    total_rewards = []
    total_success_rates = []

    for data_type in ["train", "val", "test"]:
        all_rewards = []

        # Get all the rollouts that are used to consolidate the results
        json_files = [f for f in os.listdir(os.path.join(agent_rollout_dir, data_type)) if is_valid_rollout(f)]

        # Compute rewards efficiently
        all_rewards = [sum(t["reward"] for t in load_json(os.path.join(agent_rollout_dir, data_type, f))) for f in json_files]
        mean_reward = np.mean(all_rewards)
        se_reward = np.std(all_rewards)/math.sqrt(len(all_rewards))

        # Compute success rate
        all_success_rates = [load_json(os.path.join(agent_rollout_dir, data_type, f))[-1]["reward"] == 0 for f in json_files]
        mean_success_rate = np.mean(all_success_rates)
        se_success_rate = np.std(all_success_rates)/math.sqrt(len(all_success_rates))

        if overwrite:
            table_dict[f"{data_type} (avg reward)"][table_dict["model"].index(agent_name)] = mean_reward
            table_dict[f"{data_type} (se reward)"][table_dict["model"].index(agent_name)] = se_reward
            table_dict[f"{data_type} (avg success rate)"][table_dict["model"].index(agent_name)] = mean_success_rate
            table_dict[f"{data_type} (se success rate)"][table_dict["model"].index(agent_name)] = se_success_rate
        else:
            table_dict[f"{data_type} (avg reward)"].append(mean_reward)
            table_dict[f"{data_type} (se reward)"].append(se_reward)
            table_dict[f"{data_type} (avg success rate)"].append(mean_success_rate)
            table_dict[f"{data_type} (se success rate)"].append(se_success_rate)

        total_rewards.append(mean_reward)
        total_success_rates.append(mean_success_rate)

    table_dict["total (avg reward)"].append(np.mean(total_rewards))
    table_dict["total (se reward)"].append(np.std(total_rewards)/math.sqrt(len(total_rewards)))
    table_dict["total (avg success rate)"].append(np.mean(total_success_rates))
    table_dict["total (se success rate)"].append(np.std(total_success_rates)/math.sqrt(len(total_success_rates)))

    # Convert the table dict to a dataframe and save it
    table = pd.DataFrame(table_dict)
    table.to_csv(table_fp, index=False)


@hydra.main(version_base=None, config_path="../../configs/eval_config", config_name="twenty_questions.yaml")
def main(cfg: DictConfig):
    elogger.set_activate(cfg.elogger)

    dstdir = os.path.join(cfg.logdir, f"iter{cfg.iter}")
    os.makedirs(dstdir, exist_ok=True)

    # Load the model
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

            logdir = os.path.join(dstdir, agent_name)

        if cfg.mode == "consolidate_online":
            # Saving table under a specific hydra version
            hydra_folder_path = get_output_path()
            table_fp = os.path.join(hydra_folder_path, f"online_eval_table{'_' + cfg.consolidate_online.table_notes if cfg.consolidate_online.table_notes else ''}.csv")

            consolidate_online_eval(cfg, table_fp, agent_rollout_dir=logdir, agent_name=agent_config.log_name, rollout_per_task=cfg.consolidate_online.rollout_per_task)

            # Check if the file or symlink exists, then remove it
            dst_link_fp = os.path.join(dstdir, "online_eval_table.csv")
            if os.path.exists(dst_link_fp) or os.path.islink(dst_link_fp):
                os.remove(dst_link_fp)

            # Adding a soft link to the table under dstdir
            os.symlink(table_fp, dst_link_fp)
        else:
            agent = initialize_agent(agent_config,
                                        parse_reason_action_fn=parse_reason_and_action_twenty_questions,
                                        verbose=cfg["verbose"],
                                        debug=cfg["debug"])
                 
            os.makedirs(logdir, exist_ok=True)
            print(f"Evaluating {agent_name} in {logdir}")

            if cfg.mode == "offline":
                offline_eval(cfg, logdir, agent)
            elif cfg.mode == "online":
                online_eval(cfg, logdir, agent)
            else:
                raise ValueError(f"Invalid mode: {cfg.mode}")
            
        if cfg.host_sglang:
            if processes is not None:
                for process in processes:
                    # Cleanup the SGLang server
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait()
                    print("SGLang server terminated")

    if cfg.mode == "online":
        # Because this takes a long time, we notify when the online eval is done
        elogger.log(f"Online eval results saved in {dstdir}")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        elogger.log(f"Error: {e}")
        raise e
