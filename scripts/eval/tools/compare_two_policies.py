"""
Note: policy A is typically the reference policy

Given 2 policy models, tally and store the rollouts
- where the policy A succeeds but the policy B fails
- where the policy A fails but the policy B succeeds
- where both succeed
- where both fail

Each will be stored as a separate csv file, with the following columns:
- task_id (task_name + rollout_id)
- task_category (which object category does the task belong to)
- data_type (train/val/test)
- path_to_policy_A_model
- path_to_policy_B_model
- investigation_comments

python scripts/eval/tools/compare_two_policies.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT, TEST_OBJECT_DICT
from agent_prm.envs.car_dealer.data import TRAIN_BUYER_STRATEGIES, VAL_BUYER_STRATEGIES, TEST_BUYER_STRATEGIES, TRAIN_BRANDS, VAL_BRANDS, TEST_BRANDS, TRAIN_TYPES, VAL_TYPES, TEST_TYPES
from agent_prm.utils.general_utils import load_json, save_json
from agent_prm.utils.cfg_utils import find_matching_iter

# DOMAIN = "twenty_questions"
##### Compare p0 and BoN_pi0_Q0-80pct-lr=5e-5
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-5_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "pi0"
# POLICY_B_NAME = "BoN_pi0_Q0-80pct-lr=5e-5"

##### QUESTION: Why bon_pi0_q0 worse than pi0?
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-6_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "pi0"
# POLICY_B_NAME = "BoN_pi0_Q0-80pct-lr=5e-6"

##### QUESTION: Why pi1 worse than pi0?
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-80pct_Q0-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6"
# POLICY_A_NAME = "pi0"
# POLICY_B_NAME = "pi1"

##### QUESTION: Why pi2 much better than pi0?
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi2-80pct_Q0-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6"
# POLICY_A_NAME = "pi0"
# POLICY_B_NAME = "pi2"

##### QUESTION: What's the diff with training on hindsight data?
## Baseline
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-6_hindsight-baseline_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "pi0"
# POLICY_B_NAME = "BoN_pi0_Q0-80pct-lr=5e-6_hindsight-baseline"
## Regular PRM vs Hindsight PRM
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-6_hindsight-baseline_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-6_hindsight_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "BoN_pi0_Q0-80pct-lr=5e-6_hindsight-baseline"
# POLICY_B_NAME = "BoN_pi0_Q0-80pct-lr=5e-6_hindsight"

# ##### QUESTION [Hindsight-redo]: What's the diff with training on hindsight data vs training on original data? (after we fixed the env and improved hindsight data)
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-lr=5e-6_hindsight-redo_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env"
# POLICY_B_NAME = "BoN_pi0_Q0-lr=5e-6_hindsight-redo"

##### QUESTION [Hindsight-redo]: What's the diff with training on hindsight data vs training on original data? (after we fixed the env and improved hindsight data)
## 50 pct data
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-20pct-lr=5e-6_pi0-new-env_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased-on-50_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "BoN_pi0_Q0-20pct-lr=5e-6_pi0-new-env"
# POLICY_B_NAME = "BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased-on-50"

# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-40pct_Q0-20pct-lr=5e-6_pi0-new-env_250405_214137_iter1_pi0-new-env_pi1_Q0-20pct-lr=5e-6_pi0-new-env"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-60pct_Q0-20pct-lr=5e-6_hindsight-biased-on-50_250410_004916_iter1_hindsight-biased-on-50_pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-50"
# POLICY_A_NAME = "pi1-40pct_Q0-20pct-lr=5e-6_pi0-new-env"
# POLICY_B_NAME = "pi1-60pct_Q0-20pct-lr=5e-6_hindsight-biased-on-50"

## 60 pct data
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-20pct-lr=5e-6_pi0-new-env_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased-on-60_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
# POLICY_A_NAME = "BoN_pi0_Q0-20pct-lr=5e-6_pi0-new-env"
# POLICY_B_NAME = "BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased-on-60"

## Compare the 60 pct data pi1 vs 100 pct data pi1
# POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-40pct_Q0-20pct-lr=5e-6_pi0-new-env_250405_214137_iter1_pi0-new-env_pi1_Q0-20pct-lr=5e-6_pi0-new-env"
# POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-40pct_Q0-20pct-lr=5e-6_hindsight-biased-on-60_250411_220702_iter1_hindsight-biased-on-60_pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-60"
# POLICY_A_NAME = "pi1-40pct_Q0-20pct-lr=5e-6_pi0-new-env"
# POLICY_B_NAME = "pi1-40pct_Q0-20pct-lr=5e-6_hindsight-biased-on-60"

##################
DOMAIN = "car_dealer"

POLICY_A_PATH = "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter0/pi0-83pct_250430_180817_iter0_pi0_vanilla_epochs=3"
POLICY_B_PATH = "/share/portal/hw575/agent_prm/data/car_dealer/eval/iter0/pi0-83pct_250430_180817_iter0_pi0_vanilla_epochs=3"
POLICY_A_NAME = "pi0"
POLICY_B_NAME = "pi0"
##################

save_folder_path = os.path.join(f"playground/{DOMAIN}/compare_two_policies", f"{POLICY_A_NAME}_vs_{POLICY_B_NAME}")

ROLLOUT_PER_TASK_DICT = {
    "train": 1,
    "val": 3,
    "test": 3
}

def plot_confusion_matrix(df_confusion_matrix, policy_A_name: str, policy_B_name: str, file_name="confusion_matrix.png", is_percentage=False):
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(df_confusion_matrix, cmap='Blues', interpolation='nearest')
    # Add text to the cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{df_confusion_matrix.iloc[i, j]:.2f}" if not is_percentage else f"{df_confusion_matrix.iloc[i, j]:.2f}%", ha='center', va='center', color='black', fontsize=20)
    plt.colorbar()
    plt.xticks(ticks=[0, 1], labels=[f'{policy_B_name}=0', f'{policy_B_name}=1'])
    plt.yticks(ticks=[0, 1], labels=[f'{policy_A_name}=0', f'{policy_A_name}=1'])
    plt.xlabel(f'Policy B ({policy_B_name})')
    plt.ylabel(f'Policy A ({policy_A_name})')
    plt.title(file_name)
    plt.savefig(os.path.join(save_folder_path, file_name))

def get_iterator_for_domain(domain: str, data_type: str):
    """
    List of 
        prefix, domain_specific_info_to_track
    """
    if domain == "twenty_questions":
        if data_type == "train":
            object_dict_to_use = TRAIN_OBJECT_DICT
        elif data_type == "val":
            object_dict_to_use = VALIDATION_OBJECT_DICT
        elif data_type == "test":
            object_dict_to_use = TEST_OBJECT_DICT

        game_list = [(obj, category) for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]
    elif domain == "car_dealer":
        buyer_info_dict = load_json("src/agent_prm/envs/car_dealer/buyer_info_dict.json")

        if data_type == "train":
            buyer_strategy_dict = TRAIN_BUYER_STRATEGIES
            brand_list = TRAIN_BRANDS
            type_list = TRAIN_TYPES
        elif data_type == "val":
            buyer_strategy_dict = VAL_BUYER_STRATEGIES
            brand_list = VAL_BRANDS
            type_list = VAL_TYPES
        elif data_type == "test":
            buyer_strategy_dict = TEST_BUYER_STRATEGIES
            brand_list = TEST_BRANDS
            type_list = TEST_TYPES
        
        game_list = []
        for buyer_strategy_id in buyer_strategy_dict.keys():
            for brand in brand_list:
                for car_type in type_list:
                    budget_list = buyer_info_dict[str(buyer_strategy_id)][brand][car_type].keys()
                    for budget in budget_list:
                        features = buyer_info_dict[str(buyer_strategy_id)][brand][car_type][budget]["features"]
                        game_id = f"{buyer_strategy_id}_{brand}_{car_type}_{budget}"
                        game_list.append((game_id, f"{features}:{buyer_strategy_dict[buyer_strategy_id]['name']}"))
        
    return game_list
        

def collect_results(policy_A_path: str, policy_B_path: str, policy_A_name: str, policy_B_name: str):
    os.makedirs(save_folder_path, exist_ok=True)

    meta_data_dict = {
        "policy_A_name": policy_A_name,
        "policy_A_path": policy_A_path,
        "policy_A_iter": find_matching_iter(policy_A_name),
        "policy_B_name": policy_B_name,
        "policy_B_path": policy_B_path,
        "policy_B_iter": find_matching_iter(policy_B_name),
    }
    save_json(os.path.join(save_folder_path, "meta_data.json"), meta_data_dict)

    # Initialize the csvs
    policy_A_1_policy_B_0_dict = {
        "task_id": [],
        "task_info": [],
        "data_type": [],
        "path_to_policy_A_model": [],
        "path_to_policy_B_model": [],
        "investigation_comments": []
    }
    policy_A_0_policy_B_1_dict = { 
        "task_id": [],
        "task_info": [],
        "data_type": [],
        "path_to_policy_A_model": [],
        "path_to_policy_B_model": [],
        "investigation_comments": []    
    }
    policy_A_1_policy_B_1_dict = {
        "task_id": [],
        "task_info": [],
        "data_type": [],
        "path_to_policy_A_model": [],
        "path_to_policy_B_model": [],
        "investigation_comments": []
    }
    policy_A_0_policy_B_0_dict = {
        "task_id": [],
        "task_info": [],
        "data_type": [],
        "path_to_policy_A_model": [],
        "path_to_policy_B_model": [],
        "investigation_comments": []
    }

    for data_type in ["train", "val", "test"]:
        game_list = get_iterator_for_domain(DOMAIN, data_type)

        for game_id, domain_specific_info in game_list:
            for rollout_id in range(ROLLOUT_PER_TASK_DICT[data_type]):
                policy_A_rollout_path = os.path.join(policy_A_path, data_type, f"{game_id}_{rollout_id}.json")
                policy_B_rollout_path = os.path.join(policy_B_path, data_type, f"{game_id}_{rollout_id}.json")

                policy_A_rollout = load_json(policy_A_rollout_path)
                policy_B_rollout = load_json(policy_B_rollout_path)

                if DOMAIN == "car_dealer":
                    policy_A_success = policy_A_rollout[-1]["success"]
                    policy_B_success = policy_B_rollout[-1]["success"]
                else:
                    policy_A_success = policy_A_rollout[-1]["reward"] == 0
                    policy_B_success = policy_B_rollout[-1]["reward"] == 0

                if policy_A_success and not policy_B_success:
                    dict_to_add_to = policy_A_1_policy_B_0_dict
                elif not policy_A_success and policy_B_success:
                    dict_to_add_to = policy_A_0_policy_B_1_dict
                elif policy_A_success and policy_B_success:
                    dict_to_add_to = policy_A_1_policy_B_1_dict
                elif not policy_A_success and not policy_B_success:
                    dict_to_add_to = policy_A_0_policy_B_0_dict

                dict_to_add_to["task_id"].append(f"{game_id}_{rollout_id}")
                dict_to_add_to["task_info"].append(domain_specific_info)
                dict_to_add_to["data_type"].append(data_type)
                dict_to_add_to["path_to_policy_A_model"].append(policy_A_rollout_path)
                dict_to_add_to["path_to_policy_B_model"].append(policy_B_rollout_path)
                dict_to_add_to["investigation_comments"].append("")  # Placeholder for comments

    # Save the results
    policy_A_1_policy_B_0_df = pd.DataFrame(policy_A_1_policy_B_0_dict)
    policy_A_0_policy_B_1_df = pd.DataFrame(policy_A_0_policy_B_1_dict)
    policy_A_1_policy_B_1_df = pd.DataFrame(policy_A_1_policy_B_1_dict)
    policy_A_0_policy_B_0_df = pd.DataFrame(policy_A_0_policy_B_0_dict)

    policy_A_1_policy_B_0_df.to_csv(os.path.join(save_folder_path, "policy_A_1_policy_B_0.csv"), index=False)
    policy_A_0_policy_B_1_df.to_csv(os.path.join(save_folder_path, "policy_A_0_policy_B_1.csv"), index=False)
    policy_A_1_policy_B_1_df.to_csv(os.path.join(save_folder_path, "policy_A_1_policy_B_1.csv"), index=False)
    policy_A_0_policy_B_0_df.to_csv(os.path.join(save_folder_path, "policy_A_0_policy_B_0.csv"), index=False)

    # Compute the confusion matrix (y-axis: policy A, x-axis: policy B)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = len(policy_A_0_policy_B_0_df)
    confusion_matrix[0, 1] = len(policy_A_0_policy_B_1_df)
    confusion_matrix[1, 0] = len(policy_A_1_policy_B_0_df)
    confusion_matrix[1, 1] = len(policy_A_1_policy_B_1_df)

    # Save the confusion matrix
    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=[f'{policy_A_name}=0', f'{policy_A_name}=1'], columns=[f'{policy_B_name}=0', f'{policy_B_name}=1'])
    df_confusion_matrix.to_csv(os.path.join(save_folder_path, "confusion_matrix.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix, policy_A_name, policy_B_name)

    # Compute the percentage version of the confusion matrix
    total_count = df_confusion_matrix.sum().sum()
    df_confusion_matrix_percentage = df_confusion_matrix.div(total_count).mul(100)
    df_confusion_matrix_percentage.to_csv(os.path.join(save_folder_path, "confusion_matrix_percentage.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix_percentage, policy_A_name, policy_B_name, file_name="confusion_matrix_percentage.png", is_percentage=True)


def present_per_data_type_results(policy_A_name: str, policy_B_name: str):
    # Load the results
    policy_A_1_policy_B_0_df = pd.read_csv(os.path.join(save_folder_path, "policy_A_1_policy_B_0.csv"))
    policy_A_0_policy_B_1_df = pd.read_csv(os.path.join(save_folder_path, "policy_A_0_policy_B_1.csv"))
    policy_A_1_policy_B_1_df = pd.read_csv(os.path.join(save_folder_path, "policy_A_1_policy_B_1.csv"))
    policy_A_0_policy_B_0_df = pd.read_csv(os.path.join(save_folder_path, "policy_A_0_policy_B_0.csv"))

    # Update the results based on tags in the investigation_comments
    for idx, row in policy_A_1_policy_B_0_df.iterrows():
        if type(row["investigation_comments"]) == str and ("[fail_to_detect_success]" in row["investigation_comments"] or "[sim_wrong_reply]" in row["investigation_comments"]):
            # policy_B actually succeeds, but the env fails to detect it. Move this row to policy_A_1_policy_B_1_df
            policy_A_1_policy_B_1_df = pd.concat([policy_A_1_policy_B_1_df, pd.DataFrame([row])], ignore_index=True)
            policy_A_1_policy_B_0_df = policy_A_1_policy_B_0_df.drop(idx)
    for idx, row in policy_A_0_policy_B_1_df.iterrows():
        if type(row["investigation_comments"]) == str and ("[fail_to_detect_success]" in row["investigation_comments"] or "[sim_wrong_reply]" in row["investigation_comments"]):
            # policy_B actually succeeds, but the env fails to detect it. Move this row to policy_A_1_policy_B_1_df
            policy_A_1_policy_B_1_df = pd.concat([policy_A_1_policy_B_1_df, pd.DataFrame([row])], ignore_index=True)
            policy_A_0_policy_B_1_df = policy_A_0_policy_B_1_df.drop(idx)

    # Add a column to each dataframe that indicates whether the policy succeeded or not
    policy_A_1_policy_B_0_df["policy_A_success"] = True
    policy_A_0_policy_B_1_df["policy_A_success"] = False
    policy_A_1_policy_B_1_df["policy_A_success"] = True
    policy_A_0_policy_B_0_df["policy_A_success"] = False

    # Add a column to each dataframe that indicates whether the policy B succeeded or not
    policy_A_1_policy_B_0_df["policy_B_success"] = False
    policy_A_0_policy_B_1_df["policy_B_success"] = True
    policy_A_1_policy_B_1_df["policy_B_success"] = True
    policy_A_0_policy_B_0_df["policy_B_success"] = False
            
    for data_type in ["train", "val", "test"]:
        data_type_df = pd.concat([policy_A_1_policy_B_0_df, policy_A_0_policy_B_1_df, policy_A_1_policy_B_1_df, policy_A_0_policy_B_0_df])
        data_type_df = data_type_df[data_type_df["data_type"] == data_type]

        # Compute the confusion matrix (y-axis: policy A, x-axis: policy B)
        confusion_matrix = np.zeros((2, 2))

        confusion_matrix[0, 0] = len(data_type_df[(data_type_df["policy_A_success"] == False) & (data_type_df["policy_B_success"] == False)])
        confusion_matrix[0, 1] = len(data_type_df[(data_type_df["policy_A_success"] == False) & (data_type_df["policy_B_success"] == True)])
        confusion_matrix[1, 0] = len(data_type_df[(data_type_df["policy_A_success"] == True) & (data_type_df["policy_B_success"] == False)])
        confusion_matrix[1, 1] = len(data_type_df[(data_type_df["policy_A_success"] == True) & (data_type_df["policy_B_success"] == True)])

        # Save the confusion matrix
        df_confusion_matrix = pd.DataFrame(confusion_matrix, index=[f'{policy_A_name}=0', f'{policy_A_name}=1'], columns=[f'{policy_B_name}=0', f'{policy_B_name}=1'])
        df_confusion_matrix.to_csv(os.path.join(save_folder_path, f"confusion_matrix_{data_type}_n={len(data_type_df)}.csv"), index=False)
        plot_confusion_matrix(df_confusion_matrix, policy_A_name, policy_B_name, file_name=f"confusion_matrix_{data_type}_n={len(data_type_df)}.png")

        # Compute the percentage version of the confusion matrix
        total_count = df_confusion_matrix.sum().sum()
        df_confusion_matrix_percentage = df_confusion_matrix.div(total_count).mul(100)
        df_confusion_matrix_percentage.to_csv(os.path.join(save_folder_path, f"confusion_matrix_percentage_{data_type}_n={len(data_type_df)}.csv"), index=False)
        plot_confusion_matrix(df_confusion_matrix_percentage, policy_A_name, policy_B_name, file_name=f"confusion_matrix_percentage_{data_type}_n={len(data_type_df)}.png", is_percentage=True)

    # Overall results
    data_type_df = pd.concat([policy_A_1_policy_B_0_df, policy_A_0_policy_B_1_df, policy_A_1_policy_B_1_df, policy_A_0_policy_B_0_df])

    # Compute the confusion matrix (y-axis: policy A, x-axis: policy B)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = len(data_type_df[(data_type_df["policy_A_success"] == False) & (data_type_df["policy_B_success"] == False)])
    confusion_matrix[0, 1] = len(data_type_df[(data_type_df["policy_A_success"] == False) & (data_type_df["policy_B_success"] == True)])
    confusion_matrix[1, 0] = len(data_type_df[(data_type_df["policy_A_success"] == True) & (data_type_df["policy_B_success"] == False)])
    confusion_matrix[1, 1] = len(data_type_df[(data_type_df["policy_A_success"] == True) & (data_type_df["policy_B_success"] == True)])
    
    # Save the confusion matrix
    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=[f'{policy_A_name}=0', f'{policy_A_name}=1'], columns=[f'{policy_B_name}=0', f'{policy_B_name}=1'])
    df_confusion_matrix.to_csv(os.path.join(save_folder_path, "confusion_matrix_overall.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix, policy_A_name, policy_B_name, file_name="confusion_matrix_overall.png")
    
    # Compute the percentage version of the confusion matrix
    total_count = df_confusion_matrix.sum().sum()
    df_confusion_matrix_percentage = df_confusion_matrix.div(total_count).mul(100)
    df_confusion_matrix_percentage.to_csv(os.path.join(save_folder_path, "confusion_matrix_percentage_overall.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix_percentage, policy_A_name, policy_B_name, file_name="confusion_matrix_percentage_overall.png", is_percentage=True)


if __name__ == "__main__":
    print(f"Results will be saved to {save_folder_path}")
    input("Press Enter to continue...")

    collect_results(POLICY_A_PATH, POLICY_B_PATH, POLICY_A_NAME, POLICY_B_NAME)
    present_per_data_type_results(POLICY_A_NAME, POLICY_B_NAME)