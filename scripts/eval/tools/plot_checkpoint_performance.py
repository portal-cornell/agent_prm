import yaml
import json
import re
import os
import math
import numpy as np
from matplotlib import color_sequences
import matplotlib.pyplot as plt
import argparse

from agent_prm.utils.general_utils import load_json

total_epochs = 1
baselines = ["gpt-4o", "base-3B", "pi0-lora", "pi0", "pi0_Q0*"]

folder_to_regex = {
    "sft-pi0": [r'(pi0-(\d+)pct)-all-data-3epoches', r'pi0-all-data-3epoches'],
    # "prm-pi0-q0": [r'BoN-pi0\+(\d+)pct', r'BoN-(\d+)pct-balanced-Q', r'BoN-pi0\+Q0', r'BoN-balanced-Q'],
    "prm-pi0-q0": [r'BoN_pi0_(Q0-(\d+)pct)', r'BoN_pi0_Q0-lr'],
    "rl-pi1": [r'^(pi1-(\d+)pct)_Q0-80', r'^pi1_Q0-80'],
    # 'rl-pi1-q0': [r'^pi1-(\d+)pct$', r'^pi1$', r'^BoN_pi1-(\d+)pct_Q0\*$', r'^BoN_pi1_Q0\*$']
}

folder_to_plot_models = {
    "sft-pi0": ['gpt-4o', 'base-3B', 'pi0-lora', 'pi0-all-data-3epoches'],
    # "prm-pi0-q0": ['gpt-4o', 'pi0-lora', 'pi0', 'BoN-balanced-Q-3B-PSFT-all-data-3epoches', 'BoN-balanced-Q-lr=5e-6-3B-PSFT-all-data-3epoches', 'BoN-pi0+Q0', 'BoN-pi0+Q0-lr=5e-6'],
    "prm-pi0-q0": ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-5', 'BoN_pi0_Q0-lr=5e-6', 'BoN_pi0_Q0-lr=5e-7'],
    "rl-pi1": ['gpt-4o', 'base-3B', 'pi0', 'pi1_Q0-80pct-lr=5e-6'],
    # "rl-pi1-q0": ['gpt-4o', 'pi0', 'pi0_Q0*', 'pi1', 'BoN_pi1_Q0*']
}

eval_dir = "data/twenty_questions/eval/iter0"
ROLLOUT_PER_TASK_DICT = {
    "train": 1,
    "val": 3,
    "test": 3
}

def is_valid_rollout(f: str, data_type: str) -> bool:
    """
    Check if the rollout is valid
    """
    is_a_rollout_file = f.endswith(".json") and not f.endswith("_summary_dict.json")
    to_include = False

    for i in range(ROLLOUT_PER_TASK_DICT[data_type]):
        if f"_{i}" in f:
            to_include = True
            break

    return is_a_rollout_file and to_include


def get_pct_of_training_progress(log_name: str, regex: str) -> int:
    """
    Parameters:
        log_name: str
            The log name of the model
                For example, if it's in progress: BoN-pi0+12pctQ0-lr=5e-6, 
                else if it's fully trained: BoN-pi0+Q0-lr=5e-6
    Returns:
        pct_of_training_progress: int
            The pct of training progress of the model
        class_name: str
            The class name of the model (for example: BoN-pi0+Q0-lr=5e-6)
    """
    if "pct" in regex:
        try:
            # 2 allows us to directly get the number
            pct = int(re.search(regex, log_name).group(2))
        except Exception as e:
            print(f"Error:\n{e}")
            raise ValueError(f"Invalid log name: {log_name}, or regex: {regex}")
        
        part_with_pct = re.search(regex, log_name).group(1)
        part_with_pct_removed = part_with_pct.replace(f"-{pct}pct", "")

        class_name = log_name.replace(part_with_pct, part_with_pct_removed)

        return pct, class_name
    else:
        # Assuming that it's fully trained
        return 100, log_name
    

def update_models_with_eval_results(models_to_plot: dict):
    for data_type in ["train", "val", "test"]:
        for class_name in models_to_plot:
            if class_name in baselines:
                continue

            for pct in models_to_plot[class_name]:
                agent_eval_dir = models_to_plot[class_name][pct]["agent_eval_dir"]

                # Get all the rollouts that are used to consolidate the results
                json_files = [f for f in os.listdir(os.path.join(agent_eval_dir, data_type)) if is_valid_rollout(f, data_type)]

                # Compute rewards efficiently
                all_rewards = [sum(t["reward"] for t in load_json(os.path.join(agent_eval_dir, data_type, f))) for f in json_files]
                mean_reward = np.mean(all_rewards)
                se_reward = np.std(all_rewards)/math.sqrt(len(all_rewards))

                # Compute success rate
                all_success_rates = [load_json(os.path.join(agent_eval_dir, data_type, f))[-1]["reward"] == 0 for f in json_files]
                mean_success_rate = np.mean(all_success_rates)
                se_success_rate = np.std(all_success_rates)/math.sqrt(len(all_success_rates))

                models_to_plot[class_name][pct][data_type] = {
                    "mean_reward": mean_reward,
                    "se_reward": se_reward,
                    "mean_success_rate": mean_success_rate,
                    "se_success_rate": se_success_rate
                }


def plot_models(models_to_plot: dict, plot_path: str, models_to_plot_names: list):
    """
    Plot the models and save the models at 
    """
    data_types = ['train', 'val', 'test']
    colors = color_sequences['Dark2']
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))

    linewidth = 3
    markersize = 10
    fontsize = 16

    # Plot reward metrics
    for i, data_type in enumerate(data_types):
        ax = axes[0, i]
        for j, model in enumerate(models_to_plot_names):
            if model in baselines:
                # Draw a horizontal line with standard error
                ax.axhline(y=models_to_plot[model][data_type]["mean_reward"], color=colors[j], linestyle='--', label=model, linewidth=linewidth)
            else:
                epochs = np.array([int(pct)/100.0*total_epochs for pct in models_to_plot[model].keys()])
                rewards = np.array([models_to_plot[model][pct][data_type]["mean_reward"] for pct in models_to_plot[model].keys()])
                se_rewards = np.array([models_to_plot[model][pct][data_type]["se_reward"] for pct in models_to_plot[model].keys()])

                ax.plot(epochs, rewards, label=model, color=colors[j], marker='o', linewidth=linewidth, markersize=markersize)
                ax.fill_between(epochs, rewards - se_rewards, 
                                rewards + se_rewards, color=colors[j], alpha=0.2)
            
        # Set the y axis range to be -20 and 0
        ax.set_ylim(-20, -10)
        ax.set_xticks(np.arange(0, total_epochs+total_epochs/10, total_epochs/10))
        ax.set_title(f'Reward - {data_type}', fontsize=fontsize)
        ax.set_xlabel('Epochs', fontsize=fontsize)
        ax.set_ylabel('Reward', fontsize=fontsize)

    # Plot success metrics
    for i, data_type in enumerate(data_types):
        ax = axes[1, i]
        for j, model in enumerate(models_to_plot_names):
            if model in baselines:
                # Draw a horizontal line with standard error
                ax.axhline(y=models_to_plot[model][data_type]["mean_success_rate"], color=colors[j], linestyle='--', label=model, linewidth=linewidth)
            else:
                # Calculate the epoch (x-axis)
                epochs = np.array([int(pct)/100.0*total_epochs for pct in models_to_plot[model].keys()])

                # Success rate
                success_rates = np.array([models_to_plot[model][pct][data_type]["mean_success_rate"] for pct in models_to_plot[model].keys()])
                se_success_rates = np.array([models_to_plot[model][pct][data_type]["se_success_rate"] for pct in models_to_plot[model].keys()])
                
                ax.plot(epochs, success_rates, label=model, color=colors[j], marker='o', linewidth=linewidth, markersize=markersize)
                ax.fill_between(epochs, success_rates - se_success_rates, 
                                success_rates + se_success_rates, color=colors[j], alpha=0.2)
        
        # Set the y axis range to be 0 and 1
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0, total_epochs+total_epochs/10, total_epochs/10))
        ax.set_title(f'Success - {data_type}', fontsize=fontsize)
        ax.set_xlabel('Epochs', fontsize=fontsize)
        ax.set_ylabel('Success Rate', fontsize=fontsize)


    # Add one legend for all plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fancybox=True, shadow=False, framealpha=1.0, fontsize=fontsize)

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of folder to save the plot")
    parser.add_argument("-e", "--use_existing_table", action="store_true")
    args = parser.parse_args()

    save_path = os.path.join("playground/eval", args.name)
    os.makedirs(save_path, exist_ok=True)

    try:
        regex_to_use = folder_to_regex[args.name]
        models_to_plot_names = folder_to_plot_models[args.name]
    except KeyError:
        raise ValueError(f"Invalid name: {args.name}, available names are: {folder_to_regex.keys()}")

    # Load the yaml file
    with open("configs/eval_config/twenty_questions.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not args.use_existing_table:
        # Get the models to plot
        """
        Format:
        [
            "name_of_this_class_of_models": {
                "pct_of_training_progress": "path_to_eval_dir",
                ...
            }
        ]
        """

        # Get the models to plot
        models_to_plot = {
            "gpt-4o": {
                "train": {
                    "mean_reward": -15.272727272727272,
                    "se_reward": 0.5212681756006831,
                    "mean_success_rate": 0.5181818181818182,
                    "se_success_rate": 0.0476415996113025
                },
                "val": {
                    "mean_reward": -15.857142857142858,
                    "se_reward": 0.8123948687310406,
                    "mean_success_rate": 0.5714285714285714,
                    "se_success_rate": 0.0935219529582824
                },
                "test": {
                    "mean_reward": -14.4,
                    "se_reward": 1.1562871615649808,
                    "mean_success_rate": 0.6,
                    "se_success_rate": 0.1095445115010332
                }
            },
            "base-3B": {
                "train": {
                    "mean_reward": -19.79090909090909,
                    "se_reward": 0.1029285674995681,
                    "mean_success_rate": 0.1181818181818181,
                    "se_success_rate": 0.0307799929164528
                },
                "val": {
                    "mean_reward": -19.75,
                    "se_reward": 0.0818317088384971,
                    "mean_success_rate": 0.25,
                    "se_success_rate": 0.0818317088384971
                },
                "test": {
                    "mean_reward": -19.85,
                    "se_reward": 0.0798435971133565,
                    "mean_success_rate": 0.15,
                    "se_success_rate": 0.0798435971133565
                }
            },
            "pi0-lora": {
                "train": {
                    "mean_reward": -16.681818181818183,
                    "se_reward": 0.4697338473037469,
                    "mean_success_rate": 0.3909090909090909,
                    "se_success_rate":0.0465245950159423
                },
                "val": {
                    "mean_reward": -16.571428571428573,
                    "se_reward": 0.8841411309053551,
                    "mean_success_rate": 0.4285714285714285,
                    "se_success_rate": 0.0935219529582824
                },
                "test": {
                    "mean_reward": -16.45,
                    "se_reward": 0.9911483239152452,
                    "mean_success_rate": 0.45,
                    "se_success_rate": 0.1112429773064349
                }
            },
            "pi0": {
                "train": {
                    "mean_reward": -14.627272727272729,
                    "se_reward": 0.5161617868778234,
                    "mean_success_rate": 0.6181818181818182,
                    "se_success_rate":0.0463222956185777
                },
                "val": {
                    "mean_reward": -14.535714285714286,
                    "se_reward": 1.0406068466666198,
                    "mean_success_rate": 0.5714285714285714,
                    "se_success_rate": 0.0935219529582824
                },
                "test": {
                    "mean_reward": -16.7,
                    "se_reward": 1.0933892262136111,
                    "mean_success_rate": 0.4,
                    "se_success_rate": 0.1095445115010332
                }
            },
            "pi0_Q0*": {
                "train": {
                    "mean_reward": -14.418181818181818,
                    "se_reward": 0.5583893333807142,
                    "mean_success_rate": 0.5636363636363636,
                    "se_success_rate": 0.0472854401214908
                },
                "val": {
                    "mean_reward": -13.357142857142858,
                    "se_reward": 1.0799828580866146,
                    "mean_success_rate": 0.6428571428571429,
                    "se_success_rate": 0.0905522415780553
                },
                "test": {
                    "mean_reward": -14.0,
                    "se_reward": 1.1423659658795862,
                    "mean_success_rate": 0.7,
                    "se_success_rate": 0.1024695076595959
                }
            }
        }

        for agent_config in cfg["agents"]:
            for regex in regex_to_use:
                if re.search(regex, agent_config["log_name"]):
                    pct, class_name = get_pct_of_training_progress(agent_config["log_name"], regex)

                    agent_name = agent_config["model_id"] if agent_config["type"] != "best_of_n" else agent_config["generator"]["model_id"]
                        
                    if "checkpoint" in agent_name:
                        agent_name = os.path.basename(agent_name.split("/")[-2])
                    else:
                        agent_name = os.path.basename(agent_name)

                    agent_name = f"{agent_config['log_name']}_{agent_name}"

                    agent_eval_dir = os.path.join(eval_dir, agent_name)

                    if class_name not in models_to_plot:
                        models_to_plot[class_name] = {}

                    models_to_plot[class_name][pct] = {
                        "agent_eval_dir": agent_eval_dir
                    }

        with open(os.path.join(save_path, "models_to_plot.json"), "w") as f:
            json.dump(models_to_plot, f, indent=4)

        print(json.dumps(models_to_plot, indent=4))
        input("Press Enter to continue...")

        print("Updating models with eval results")
        update_models_with_eval_results(models_to_plot)

        with open(os.path.join(save_path, "models_to_plot_with_eval_results.json"), "w") as f:
            json.dump(models_to_plot, f, indent=4)
    else:
        with open(os.path.join(save_path, "models_to_plot_with_eval_results.json"), "r") as f:
            models_to_plot = json.load(f)

    print("Plotting models")
    plot_models(models_to_plot, os.path.join(save_path, f"{args.name}_plot.png"), models_to_plot_names)