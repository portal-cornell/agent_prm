"""
For each hindsight rollout, we have to find the corresponding original rollout that has the summary. Copy that summary to the hindsight rollout.

python scripts/dataproc/tools/add_summary_to_hindsight_rollouts.py
"""
import os
import json
from tqdm import tqdm

from agent_prm.envs.twenty_questions.data import TRAIN_OBJECT_DICT, VALIDATION_OBJECT_DICT

dir_path = "REDACTED"


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

for data_type in ["train", "val"]:
    object_dict_to_use = TRAIN_OBJECT_DICT if data_type == "train" else VALIDATION_OBJECT_DICT
    objects_to_eval_on = [obj for category in object_dict_to_use.keys() for obj in object_dict_to_use[category]]

    iteration_range = [
        [(0, 4), (4, 12)],
        [(20, 24), (24, 32)]
    ]

    for iteration in range(len(iteration_range)):
        policy_rollout_range = iteration_range[iteration][0]
        expert_rollout_range = iteration_range[iteration][1]
        for obj in tqdm(objects_to_eval_on):
            policy_rollouts = []
            for rollout_idx in range(policy_rollout_range[0], policy_rollout_range[1]):
                rollout_dir = os.path.join(dir_path, data_type, f"{obj}_{rollout_idx}.json")

                if not os.path.exists(rollout_dir):
                    print(f"Policy rollout {rollout_dir} does not exist")
                    continue

                with open(rollout_dir, "r") as f:
                    rollout_data = json.load(f)
                policy_rollouts.append(rollout_data)

            for rollout_idx in range(expert_rollout_range[0], expert_rollout_range[1]):
                rollout_dir = os.path.join(dir_path, data_type, f"{obj}_{rollout_idx}.json")
                # print(f"Checking for expert rollout {rollout_dir}")
                
                if not os.path.exists(rollout_dir):
                    print(f"Expert rollout {rollout_dir} does not exist")
                    continue

                with open(rollout_dir, "r") as f:
                    expert_rollout_data = json.load(f)

                raw_policy_rollout_idx, pct_covered = check_for_original_policy_rollout(expert_rollout_data, policy_rollouts)
                policy_rollout_idx = policy_rollout_range[0] + raw_policy_rollout_idx

                if raw_policy_rollout_idx == -1:
                    print(f"No matching policy rollout found for {rollout_dir}")
                    continue

                # Copy the summary to the hindsight rollout
                expert_rollout_data[0]["summary"] = policy_rollouts[raw_policy_rollout_idx][0]["summary"]

                print(f"copying summary from {policy_rollout_idx} to {rollout_idx}")

                # Save the hindsight rollout
                with open(rollout_dir, "w") as f:
                    json.dump(expert_rollout_data, f, indent=4)
