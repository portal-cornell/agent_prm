"""
Given a directory, count the number of success and failure after expert did labeling

data range from 
0-3: (4 rollouts) From the original policy
4-11: From the expert (4 * 2) rollouts
12-19: From the expert (4 * 2) rollouts (randomly selected some rollouts from 4-11, before getting expert rollouts)

python scripts/dataproc/tools/analyze_hindsight_rollout.py -d train
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_type", type=str, required=True)
parser.add_argument("-e", "--use_existing_json", default=False, action="store_true", help="Whether to skip the rollout checking process and use the existing json file")
args = parser.parse_args()

BASE_PATH = "playground/twenty_questions/hindsight"

# FOLDER_NAME = "inspect_iter1_hindsight-redo_data"
# data_iter = "iter0"
# dir_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/hindsight-redo_pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"

# FOLDER_NAME = "inspect_iter2_hindsight-biased-on-60_data"
# data_iter = "iter1"
# dir_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60_250419_134330_iter1_hindsight-biased-on-60_pi1_Q0-lr=5e-6_hindsight-biased-on-60"

# FOLDER_NAME = "inspect_iter3_hindsight-biased-on-60_data"
# data_iter = "iter2"
# dir_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter2/pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0_250423_205810_iter2_hindsight-biased-on-60_pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0"

# FOLDER_NAME = "inspect_iter1_best-pi-relabel_data"
# data_iter = "iter0"
# dir_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/best-pi-relabel_pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"

# FOLDER_NAME = "inspect_iter1_explorative-pi-relabel_data"
# data_iter = "iter0"
# dir_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/explorative-pi-relabel_pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"

FOLDER_NAME = "inspect_iter0_high-temp-pi-relabel_data"
data_iter = "iter0"
dir_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/high-temp-pi-relabel_pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all"

os.makedirs(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{args.data_type}"), exist_ok=True)

def check_for_original_policy_rollout(expert_rollout_data, policy_rollouts):
    """
    Return:
        index of the policy rollout in policy_rollouts that is the prefix of the expert rollout
        pct of expert rollout that is covered by the policy rollout
    """
    # Identify when the expert rollout starts
    expert_rollout_start_idx = -1
    for i in range(len(expert_rollout_data)):
        if expert_rollout_data[i]["raw_text"] == "":
            expert_rollout_start_idx = i

    # print(f"Expert rollout start idx: {expert_rollout_start_idx}")

    for i in range(len(policy_rollouts)):
        policy_rollout = policy_rollouts[i]
        is_prefix = True
        check = False
        for j in range(min(len(policy_rollout), expert_rollout_start_idx)):
            if policy_rollout[j]["reason"] != expert_rollout_data[j]["reason"]:
                is_prefix = False
                break
            
            if policy_rollout[j]["alternatives"] is not None and expert_rollout_data[j]["alternatives"] is not None:
                for k in range(len(policy_rollout[j]["alternatives"])):
                    if policy_rollout[j]["alternatives"][k]["reason"] != expert_rollout_data[j]["alternatives"][k]["reason"]:
                        is_prefix = False
                        break
            else:
                check = True
                # print(f"Policy rollout {i} at idx {j}:\n{policy_rollout[j]}")
                # input("sotp")
        
        if is_prefix:
            # print(f"Found matching policy rollout {i}")
            # if check:
            #     print(f'policy_rollout: {len(policy_rollout)}' + '-' * 50)
            #     print(json.dumps(policy_rollout[:expert_rollout_start_idx], indent=4))
            #     print(f"expert_rollout_start_idx: {expert_rollout_start_idx}, expert_rollout: {len(expert_rollout_data)}" + '-' * 100)
            #     print(json.dumps(expert_rollout_data[:expert_rollout_start_idx], indent=4))
            #     input("check")
            return i, expert_rollout_start_idx/len(policy_rollout)
    
    return -1, -1

def plot_confusion_matrix(df_confusion_matrix, policy_A_name: str, policy_B_name: str, save_folder_path: str, file_name="confusion_matrix.png", is_percentage=False):
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

data_type = args.data_type

object_dict_to_use = TRAIN_OBJECT_DICT if data_type == "train" else VALIDATION_OBJECT_DICT
objects_to_eval_on = [obj for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]

# Step 1: assert that all objects have the same number of files
do_initial_check = input("Do initial check? (y/n): ").lower() == "y"

if do_initial_check:
    file_per_object = {obj: [] for obj in objects_to_eval_on}

    for obj in objects_to_eval_on:
        for file in os.listdir(os.path.join(dir_path, data_type)):
            if f"{obj}_" in file and int(file.split("_")[-1].split(".")[0]) < 32:
                file_per_object[obj].append(file)

    # most common number of files
    num_files_per_object = [len(file_per_object[obj]) for obj in objects_to_eval_on]
    # mode of the number of files per object
    max_num_files = max(num_files_per_object)
    print(f"Most common number of files per object: {max_num_files}")

    # assert all the objects exist in the directory
    objs_not_in_dir = [obj for obj in objects_to_eval_on if obj not in file_per_object]
    assert objs_not_in_dir == [], f"Objects not in directory: {objs_not_in_dir}"

    # assert that all objects have the same number of files
    obj_missing_files = []
    for obj in objects_to_eval_on:
        if len(file_per_object[obj]) != max_num_files:
            print(f"Object {obj} has {len(file_per_object[obj])} files: {file_per_object[obj]}")
            obj_missing_files.append((obj, len(file_per_object[obj]), file_per_object[obj]))

    if len(obj_missing_files) > 0:
        print(f"Objects missing files: {obj_missing_files}")
        assert obj_missing_files == [], "There are objects that do not have the same number of files"

    # Total number of files in the directory
    total_num_files = sum(num_files_per_object)
    print(f"Number of objects: {len(objects_to_eval_on)}")
    print(f"Total number of files in the directory: {total_num_files}")

    input("Press Enter to continue...")

iteration_range = [
    [(0, 4), (4, 12)],
    [(20, 24), (24, 32)]
]

rollout_idx_list = []
for policy_rollout_range, expert_rollout_range in iteration_range:
    rollout_idx_list.extend(list(range(policy_rollout_range[0], policy_rollout_range[1])))
    rollout_idx_list.extend(list(range(expert_rollout_range[0], expert_rollout_range[1])))
rollout_idx_list = sorted(list(set(rollout_idx_list)))
print(f"Rollout idx list: {rollout_idx_list}")
input("Press Enter to continue...")

if not args.use_existing_json:
    policy_2_expert_dict = {iteration: {obj: {rollout_idx: [] for rollout_idx in rollout_idx_list} for obj in objects_to_eval_on} for iteration in range(len(iteration_range))}  # Keeps track of which policy rollouts are used to branch out expert rollouts

    policy_rollout_success_count = 0
    policy_total_count = 0
    expert_rollout_success_count = 0
    expert_total_count = 0

    for iteration in range(len(iteration_range)):
        policy_rollout_range = iteration_range[iteration][0]
        expert_rollout_range = iteration_range[iteration][1]
        for obj in objects_to_eval_on:
            policy_rollouts = []
            policy_rollouts_success_list = []
            for rollout_idx in range(policy_rollout_range[0], policy_rollout_range[1]):
                rollout_dir = os.path.join(dir_path, data_type, f"{obj}_{rollout_idx}.json")

                if not os.path.exists(rollout_dir):
                    print(f"Policy rollout {rollout_dir} does not exist")
                    continue

                with open(rollout_dir, "r") as f:
                    rollout_data = json.load(f)
                policy_rollouts.append(rollout_data)
                success = 1 if rollout_data[-1]["reward"] == 0 else 0
                policy_rollouts_success_list.append(success)
                policy_rollout_success_count += success
                policy_total_count += 1

            for rollout_idx in range(expert_rollout_range[0], expert_rollout_range[1]):
                rollout_dir = os.path.join(dir_path, data_type, f"{obj}_{rollout_idx}.json")
                # print(f"Checking for expert rollout {rollout_dir}")
                
                if not os.path.exists(rollout_dir):
                    print(f"Expert rollout {rollout_dir} does not exist")
                    continue

                with open(rollout_dir, "r") as f:
                    expert_rollout_data = json.load(f)
                success = 1 if expert_rollout_data[-1]["reward"] == 0 else 0
                expert_rollout_success_count += success
                expert_total_count += 1

                raw_policy_rollout_idx, pct_covered = check_for_original_policy_rollout(expert_rollout_data, policy_rollouts)
                policy_rollout_idx = policy_rollout_range[0] + raw_policy_rollout_idx
                # print(f"iteration: {iteration}, obj: {obj}, policy_rollout_idx: {policy_rollout_idx}")
                if policy_rollout_idx != -1:
                    policy_2_expert_dict[iteration][obj][policy_rollout_idx].append({
                        "expert_path": rollout_dir,
                        "pct_covered": pct_covered,
                        "policy_0_expert_0": int(policy_rollouts_success_list[raw_policy_rollout_idx] == 0 and success == 0),
                        "policy_0_expert_1": int(policy_rollouts_success_list[raw_policy_rollout_idx] == 0 and success == 1),
                        "policy_1_expert_0": int(policy_rollouts_success_list[raw_policy_rollout_idx] == 1 and success == 0),
                        "policy_1_expert_1": int(policy_rollouts_success_list[raw_policy_rollout_idx] == 1 and success == 1),
                    })

    print(f"Policy success count: {policy_rollout_success_count}/{policy_total_count}")
    print(f"Expert success count: {expert_rollout_success_count}/{expert_total_count}")

    with open(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}", f"policy_2_expert_dict_{data_type}.json"), "w") as f:
        json.dump(policy_2_expert_dict, f, indent=4)

with open(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}", f"policy_2_expert_dict_{data_type}.json"), "r") as f:
    policy_2_expert_dict = json.load(f)

# Compile a csv of the policy and expert pair where the policy = 1 but the expert = 0
policy_1_expert_0_pair = {
    "task_id": [],
    "task_category": [],
    "data_type": [],
    "path_to_policy_A_model": [], # policy
    "path_to_policy_B_model": [], # expert
    "investigation_comments": [],
}

policy_0_expert_1_pair = {
    "task_id": [],
    "task_category": [],
    "data_type": [],
    "path_to_policy_A_model": [], # policy
    "path_to_policy_B_model": [], # expert
    "investigation_comments": []
}

pct_covered_dict = {
    "policy_0_expert_0": [],
    "policy_0_expert_1": [],
    "policy_1_expert_0": [],
    "policy_1_expert_1": [],
}

# Compute the confusion matrix (y-axis: policy, x-axis: expert)
confusion_matrix = np.zeros((2, 2))

for iteration in range(len(iteration_range)):
    for obj in objects_to_eval_on:
        for rollout_idx in rollout_idx_list:
            iteration_str = str(iteration)
            rollout_idx_str = str(rollout_idx)
            if policy_2_expert_dict[iteration_str][obj][rollout_idx_str] != []:
                for policy_2_expert_dict_item in policy_2_expert_dict[iteration_str][obj][rollout_idx_str]:
                    confusion_matrix[0, 0] += policy_2_expert_dict_item["policy_0_expert_0"]
                    confusion_matrix[0, 1] += policy_2_expert_dict_item["policy_0_expert_1"]
                    confusion_matrix[1, 0] += policy_2_expert_dict_item["policy_1_expert_0"]
                    confusion_matrix[1, 1] += policy_2_expert_dict_item["policy_1_expert_1"]

                    dict_to_append_to = None
                    if policy_2_expert_dict_item["policy_1_expert_0"] == 1:
                        dict_to_append_to = policy_1_expert_0_pair
                        pct_covered_dict["policy_1_expert_0"].append(policy_2_expert_dict_item["pct_covered"])
                    elif policy_2_expert_dict_item["policy_0_expert_1"] == 1:
                        dict_to_append_to = policy_0_expert_1_pair
                        pct_covered_dict["policy_0_expert_1"].append(policy_2_expert_dict_item["pct_covered"])
                    elif policy_2_expert_dict_item["policy_0_expert_0"] == 1:
                        pct_covered_dict["policy_0_expert_0"].append(policy_2_expert_dict_item["pct_covered"])
                    elif policy_2_expert_dict_item["policy_1_expert_1"] == 1:
                        pct_covered_dict["policy_1_expert_1"].append(policy_2_expert_dict_item["pct_covered"])
                    
                    if dict_to_append_to is not None:
                        dict_to_append_to["task_id"].append(obj + "_" + rollout_idx_str)
                        dict_to_append_to["task_category"].append(obj)
                        dict_to_append_to["data_type"].append(data_type)
                        policy_path = os.path.join(dir_path, data_type, f"{obj}_{rollout_idx}.json")
                        dict_to_append_to["path_to_policy_A_model"].append(policy_path)
                        dict_to_append_to["path_to_policy_B_model"].append(policy_2_expert_dict_item["expert_path"])
                        dict_to_append_to["investigation_comments"].append("")

# Save the confusion matrix
df_confusion_matrix = pd.DataFrame(confusion_matrix, index=['policy=0', 'policy=1'], columns=['expert=0', 'expert=1'])
df_confusion_matrix.to_csv(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}", f"confusion_matrix_{data_type}.csv"))
plot_confusion_matrix(df_confusion_matrix, "policy", "expert", os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}"), f"confusion_matrix_{data_type}.png", is_percentage=False)

# Save the policy and expert pair as a csv
df_policy_1_expert_0_pair = pd.DataFrame(policy_1_expert_0_pair)
# Naming it policy_A_1_policy_B_0_{data_type}.csv to match the data viewer
df_policy_1_expert_0_pair.to_csv(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}", f"policy_A_1_policy_B_0.csv"), index=False)

df_policy_0_expert_1_pair = pd.DataFrame(policy_0_expert_1_pair)
df_policy_0_expert_1_pair.to_csv(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}", f"policy_A_0_policy_B_1.csv"), index=False)

# Calculate the avg and std of the pct_covered
for key in pct_covered_dict.keys():
    print(f"{key} (mean, std): {np.mean(pct_covered_dict[key]):.2f}, {np.std(pct_covered_dict[key]):.2f}")
    pct_covered_dict[key] = [np.mean(pct_covered_dict[key]), np.std(pct_covered_dict[key])]

df_pct_covered_dict = pd.DataFrame(pct_covered_dict)
df_pct_covered_dict.to_csv(os.path.join(BASE_PATH, FOLDER_NAME, f"{FOLDER_NAME}_{data_type}", f"pct_covered_dict_{data_type}.csv"), index=["mean", "std"])

# Meta data (used by the data_viewer)
meta_data = {
    "policy_A_name": "original_policy",
    "policy_A_path": dir_path,
    "policy_A_iter": data_iter,
    "policy_B_name": "expert_augmented_policy",
    "policy_B_path": dir_path,
    "policy_B_iter": data_iter,
}
