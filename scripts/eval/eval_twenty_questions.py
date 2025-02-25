import os
import time
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
from typing import List, Dict
import numpy as np
import pandas as pd

from agent_prm.agents.agent_registry import initialize_agent
from agent_prm.agents.agent import Agent
from agent_prm.utils.parser import parse_reason_and_action_twenty_questions
from agent_prm.utils.cfg_utils import get_output_folder_name
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT, TEST_OBJECT_DICT, WordVariants, get_default_word_list
from agent_prm.envs.twenty_questions.env import setup_twenty_questions_env
from agent_prm.utils.logger_email import elogger

def offline_eval(cfg: dict, agent: Agent):
    """
    Load the dataset and evaluate the model
    """
    pass


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


def online_eval(cfg: dict, logdir: str, agent: Agent):
    """
    Evaluate the model by interacting with the environment
    """
    rollout_per_obj = cfg.rollout_per_task
    env = setup_twenty_questions_env()
    all_obj_list = [wv[0] for wv in get_default_word_list("all")]

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

            for category in object_dict_to_use.keys():
                for obj in object_dict_to_use[category]:
                    if obj in summary_dict[rollout_idx_str]:
                        print(f"Skipping {obj} as it is already in the summary dict")
                        continue
                    obj_to_process = WordVariants.from_str(obj)
                    
                    history = env.reset(word=obj_to_process)
                    done = False
                    total_reward = 0.0
                    traj_list = []

                    while not done:
                        last_question = len(history) == env.max_conversation_length - 1

                        reason, action = query_agent(agent, history, all_obj_list, last_question)

                        print(f"++++++ agent step: {len(history)} ++++++")
                        print(f"Reason:\n{reason}\nAction:\n{action}", )
                        print(f"++++++ agent step: {len(history)} ++++++")

                        obs, reward, done = env.step(history, action)
                        history, answerer_reason, answer = obs
                        total_reward += reward

                        traj_list.append({
                            "reason": reason,
                            "question": action,
                            "answerer_reason": answerer_reason,
                            "answer": answer,
                            "reward": reward,
                        })

                    summary_dict[rollout_idx_str].append(obj)

                    save_json(summary_dict_fp, summary_dict)
                    save_json(os.path.join(logdir, data_type, f"{obj}_{rollout_idx_str}.json"), traj_list)

                    print(f"======== collected idx={rollout_idx_str} obj={obj} with total reward {total_reward}")


def consolidate_online_eval(cfg: dict, logdir: str, agent_rollout_dir: str, agent_name: str):
    """
    Consolidate the online eval results and save it as a csv file
    """
    table_fp = os.path.join(logdir, "online_eval_table.csv")

    if not os.path.exists(table_fp):
        table_dict = {
            "model": [],
            "train (avg reward)": [],
            "train (std reward)": [],
            "val (avg reward)": [],
            "val (std reward)": [],
            "test (avg reward)": [],
            "test (std reward)": []
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

    for data_type in cfg.data_types:
        all_rewards = []

        # Get all the json files that's not _summary_dict.json
        json_files = [f for f in os.listdir(os.path.join(agent_rollout_dir, data_type)) if f.endswith(".json") and not f.endswith("_summary_dict.json")]

        # Compute rewards efficiently
        all_rewards = [sum(t["reward"] for t in load_json(os.path.join(agent_rollout_dir, data_type, f))) for f in json_files if "_0" in f]

        # Update the table dict
        if overwrite:
            table_dict[f"{data_type} (avg reward)"][table_dict["model"].index(agent_name)] = np.mean(all_rewards)
            table_dict[f"{data_type} (std reward)"][table_dict["model"].index(agent_name)] = np.std(all_rewards)
        else:
            table_dict[f"{data_type} (avg reward)"].append(np.mean(all_rewards))
            table_dict[f"{data_type} (std reward)"].append(np.std(all_rewards))

    # Convert the table dict to a dataframe and save it
    table = pd.DataFrame(table_dict)
    table.to_csv(table_fp, index=False)



@hydra.main(version_base=None, config_path="../../configs/eval_config", config_name="20questions.yaml")
def main(cfg: DictConfig):
    elogger.set_activate(cfg.elogger)

    dstdir = os.path.join(cfg.logdir, f"iter{cfg.iter}")
    os.makedirs(dstdir, exist_ok=True)

    # Load the model
    for agent_config in cfg.agents:
        if agent_config.type == "gpt4o_expert":
            # Use the data collected for SFT
            assert cfg.mode == "consolidate_online", "Gpt4o expert can only be used in consolidate_online mode, where we are comparing the performance of different models"
            logdir = os.path.join(cfg.rollout_data_dir, f"iter{cfg.iter}")
            agent_name = "gpt4o_expert"
        else:
            agent = initialize_agent(agent_config,
                                        parse_reason_action_fn=parse_reason_and_action_twenty_questions,
                                        verbose=cfg["verbose"],
                                        debug=cfg["debug"])
            
            if "checkpoint" in agent.name():
                agent_name = os.path.basename(agent.name().split("/")[-2])
            else:
                agent_name = os.path.basename(agent.name())

            agent_name = f"{agent_config.log_name}_{agent_name}"

            logdir = os.path.join(dstdir, agent_name)

        if cfg.mode == "consolidate_online":
            consolidate_online_eval(cfg, dstdir, agent_rollout_dir=logdir, agent_name=agent_config.log_name)
        else:
            os.makedirs(logdir, exist_ok=True)
            print(f"Evaluating {agent_name} in {logdir}")

            if cfg.mode == "offline":
                offline_eval(cfg, logdir, agent)
            elif cfg.mode == "online":
                online_eval(cfg, logdir, agent)
            else:
                raise ValueError(f"Invalid mode: {cfg.mode}")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        elogger.log(f"Error: {e}")
        raise e
