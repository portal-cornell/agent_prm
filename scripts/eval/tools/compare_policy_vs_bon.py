"""
Given 2 policy models, tally and store the rollouts
- where the policy succeeds but the BoN fails
- where the policy fails but the BoN succeeds
- where both succeed
- where both fail

Each will be stored as a separate csv file, with the following columns:
- task_id (task_name + rollout_id)
- task_category (which object category does the task belong to)
- data_type (train/val/test)
- path_to_policy_model
- path_to_bon_model
- investigation_comments
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT, TEST_OBJECT_DICT
from agent_prm.utils.general_utils import load_json

AGENT_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"
BON_PATH = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-5_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"

AGENT_NAME = "pi0"
BON_NAME = "BoN_pi0_Q0-80pct-lr=5e-5"

save_folder_path = os.path.join("playground/compare_policy_vs_bon", f"{AGENT_NAME}_vs_{BON_NAME}")

def plot_confusion_matrix(df_confusion_matrix, file_name="confusion_matrix.png", is_percentage=False):
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(df_confusion_matrix, cmap='Blues', interpolation='nearest')
    # Add text to the cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{df_confusion_matrix.iloc[i, j]:.2f}" if not is_percentage else f"{df_confusion_matrix.iloc[i, j]:.2f}%", ha='center', va='center', color='black', fontsize=20)
    plt.colorbar()
    plt.xticks(ticks=[0, 1], labels=['bon_0', 'bon_1'])
    plt.yticks(ticks=[0, 1], labels=['policy_0', 'policy_1'])
    plt.xlabel('BoN')
    plt.ylabel('Policy')
    plt.title(file_name)
    plt.savefig(os.path.join(save_folder_path, file_name))


def collect_results(agent_path: str, bon_path: str):
    os.makedirs(save_folder_path, exist_ok=True)

    # Initialize the csvs
    policy_1_bon_0_dict = {
        "task_id": [],
        "task_category": [],
        "data_type": [],
        "path_to_policy_model": [],
        "path_to_bon_model": [],
        "investigation_comments": []
    }
    policy_0_bon_1_dict = { 
        "task_id": [],
        "task_category": [],
        "data_type": [],
        "path_to_policy_model": [],
        "path_to_bon_model": [],
        "investigation_comments": []    
    }
    policy_1_bon_1_dict = {
        "task_id": [],
        "task_category": [],
        "data_type": [],
        "path_to_policy_model": [],
        "path_to_bon_model": [],
        "investigation_comments": []
    }
    policy_0_bon_0_dict = {
        "task_id": [],
        "task_category": [],
        "data_type": [],
        "path_to_policy_model": [],
        "path_to_bon_model": [],
        "investigation_comments": []
    }

    for data_type in ["train", "val", "test"]:
        if data_type == "train":
            object_dict_to_use = TRAIN_OBJECT_DICT
        elif data_type == "val":
            object_dict_to_use = VALIDATION_OBJECT_DICT
        elif data_type == "test":
            object_dict_to_use = TEST_OBJECT_DICT

        objs_list = [(obj, data_type, category) for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]

        for obj, data_type, category in objs_list:
            agent_rollout_path = os.path.join(agent_path, data_type, f"{obj}_0.json")
            bon_rollout_path = os.path.join(bon_path, data_type, f"{obj}_0.json")

            agent_rollout = load_json(agent_rollout_path)
            bon_rollout = load_json(bon_rollout_path)

            agent_success = agent_rollout[-1]["reward"] == 0
            bon_success = bon_rollout[-1]["reward"] == 0

            if agent_success and not bon_success:
                dict_to_add_to = policy_1_bon_0_dict
            elif not agent_success and bon_success:
                dict_to_add_to = policy_0_bon_1_dict
            elif agent_success and bon_success:
                dict_to_add_to = policy_1_bon_1_dict
            elif not agent_success and not bon_success:
                dict_to_add_to = policy_0_bon_0_dict

            dict_to_add_to["task_id"].append(f"{obj}_0")
            dict_to_add_to["task_category"].append(category)
            dict_to_add_to["data_type"].append(data_type)
            dict_to_add_to["path_to_policy_model"].append(agent_rollout_path)
            dict_to_add_to["path_to_bon_model"].append(bon_rollout_path)
            dict_to_add_to["investigation_comments"].append("")  # Placeholder for comments

    # Save the results
    policy_1_bon_0_df = pd.DataFrame(policy_1_bon_0_dict)
    policy_0_bon_1_df = pd.DataFrame(policy_0_bon_1_dict)
    policy_1_bon_1_df = pd.DataFrame(policy_1_bon_1_dict)
    policy_0_bon_0_df = pd.DataFrame(policy_0_bon_0_dict)

    policy_1_bon_0_df.to_csv(os.path.join(save_folder_path, "policy_1_bon_0.csv"), index=False)
    policy_0_bon_1_df.to_csv(os.path.join(save_folder_path, "policy_0_bon_1.csv"), index=False)
    policy_1_bon_1_df.to_csv(os.path.join(save_folder_path, "policy_1_bon_1.csv"), index=False)
    policy_0_bon_0_df.to_csv(os.path.join(save_folder_path, "policy_0_bon_0.csv"), index=False)

    # Compute the confusion matrix (y-axis: policy, x-axis: bon)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = len(policy_0_bon_0_df)
    confusion_matrix[0, 1] = len(policy_0_bon_1_df)
    confusion_matrix[1, 0] = len(policy_1_bon_0_df)
    confusion_matrix[1, 1] = len(policy_1_bon_1_df)

    # Save the confusion matrix
    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=['policy_0', 'policy_1'], columns=['bon_0', 'bon_1'])
    df_confusion_matrix.to_csv(os.path.join(save_folder_path, "confusion_matrix.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix)

    # Compute the percentage version of the confusion matrix
    total_count = df_confusion_matrix.sum().sum()
    df_confusion_matrix_percentage = df_confusion_matrix.div(total_count).mul(100)
    df_confusion_matrix_percentage.to_csv(os.path.join(save_folder_path, "confusion_matrix_percentage.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix_percentage, file_name="confusion_matrix_percentage.png", is_percentage=True)


def present_per_data_type_results():
    # Load the results
    policy_1_bon_0_df = pd.read_csv(os.path.join(save_folder_path, "policy_1_bon_0.csv"))
    policy_0_bon_1_df = pd.read_csv(os.path.join(save_folder_path, "policy_0_bon_1.csv"))
    policy_1_bon_1_df = pd.read_csv(os.path.join(save_folder_path, "policy_1_bon_1.csv"))
    policy_0_bon_0_df = pd.read_csv(os.path.join(save_folder_path, "policy_0_bon_0.csv"))

    # Update the results based on tags in the investigation_comments
    for idx, row in policy_1_bon_0_df.iterrows():
        if type(row["investigation_comments"]) == str and "[fail_to_detect_success]" in row["investigation_comments"]:
            # BoN actually succeeds, but the env fails to detect it. Move this row to policy_1_bon_1_df
            policy_1_bon_1_df = pd.concat([policy_1_bon_1_df, pd.DataFrame([row])], ignore_index=True)
            policy_1_bon_0_df = policy_1_bon_0_df.drop(idx)

    # Add a column to each dataframe that indicates whether the policy succeeded or not
    policy_1_bon_0_df["policy_success"] = True
    policy_0_bon_1_df["policy_success"] = False
    policy_1_bon_1_df["policy_success"] = True
    policy_0_bon_0_df["policy_success"] = False

    # Add a column to each dataframe that indicates whether the BoN succeeded or not
    policy_1_bon_0_df["bon_success"] = False
    policy_0_bon_1_df["bon_success"] = True
    policy_1_bon_1_df["bon_success"] = True
    policy_0_bon_0_df["bon_success"] = False
            
    for data_type in ["train", "val", "test"]:
        data_type_df = pd.concat([policy_1_bon_0_df, policy_0_bon_1_df, policy_1_bon_1_df, policy_0_bon_0_df])
        data_type_df = data_type_df[data_type_df["data_type"] == data_type]

        # Compute the confusion matrix (y-axis: policy, x-axis: bon)
        confusion_matrix = np.zeros((2, 2))

        confusion_matrix[0, 0] = len(data_type_df[(data_type_df["policy_success"] == False) & (data_type_df["bon_success"] == False)])
        confusion_matrix[0, 1] = len(data_type_df[(data_type_df["policy_success"] == False) & (data_type_df["bon_success"] == True)])
        confusion_matrix[1, 0] = len(data_type_df[(data_type_df["policy_success"] == True) & (data_type_df["bon_success"] == False)])
        confusion_matrix[1, 1] = len(data_type_df[(data_type_df["policy_success"] == True) & (data_type_df["bon_success"] == True)])

        # Save the confusion matrix
        df_confusion_matrix = pd.DataFrame(confusion_matrix, index=['policy_0', 'policy_1'], columns=['bon_0', 'bon_1'])
        df_confusion_matrix.to_csv(os.path.join(save_folder_path, f"confusion_matrix_{data_type}_n={len(data_type_df)}.csv"), index=False)
        plot_confusion_matrix(df_confusion_matrix, file_name=f"confusion_matrix_{data_type}_n={len(data_type_df)}.png")

        # Compute the percentage version of the confusion matrix
        total_count = df_confusion_matrix.sum().sum()
        df_confusion_matrix_percentage = df_confusion_matrix.div(total_count).mul(100)
        df_confusion_matrix_percentage.to_csv(os.path.join(save_folder_path, f"confusion_matrix_percentage_{data_type}_n={len(data_type_df)}.csv"), index=False)
        plot_confusion_matrix(df_confusion_matrix_percentage, file_name=f"confusion_matrix_percentage_{data_type}_n={len(data_type_df)}.png", is_percentage=True)


if __name__ == "__main__":
    # collect_results(AGENT_PATH, BON_PATH)
    present_per_data_type_results()