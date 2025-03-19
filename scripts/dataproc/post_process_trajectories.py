import os
import json
import shutil
from datetime import datetime
import pandas as pd
from agent_prm.envs.twenty_questions.data import is_done, WordVariants

eval_folder_list = [
    # "/share/portal/hw575/agent_prm/data/twenty_questions/rollout/iter0", # gpt4o
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/baseline/3B_Llama-3.2-3B-Instruct", # 3B
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/pi0-all-data-3epoches_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all", # pi0
    # BoNs pi0
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-20pct-lr=5e-6_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-40pct-lr=5e-6_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-60pct-lr=5e-6_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-80pct-lr=5e-6_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/BoN_pi0_Q0-lr=5e-6_250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all",
    # pi1
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1_Q0-80pct-lr=5e-6_online_dpo",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-20pct_Q0-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-40pct_Q0-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-60pct_Q0-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    # "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/pi1-80pct_Q0-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    # BoNs pi1 q1
    "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/BoN_pi1_Q1-20pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/BoN_pi1_Q1-40pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/BoN_pi1_Q1-60pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/BoN_pi1_Q1-80pct-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6",
    "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter1/BoN_pi1_Q1-lr=5e-6_250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6"
]

curr_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
def is_valid_rollout(f: str) -> bool:
    """
    Check if the rollout is valid
    """
    return f.endswith(".json") and not f.endswith("_original.json") and not f.endswith("_summary_dict.json")

for eval_folder in eval_folder_list:
    print(f"Processing {eval_folder}")

    # Keeps track of all the edits
    edit_dict = {
        "data_type": [],
        "json_file": [],
        "t": [],
        "original_action": [],
        "original_env_answer": [],
    }

    for data_type in ["train", "val", "test"]:
        # Get all the rollouts that are used to consolidate the results
        json_files = [f for f in os.listdir(os.path.join(eval_folder, data_type)) if is_valid_rollout(f)]

        print(f"Found {len(json_files)} rollouts for {data_type}")

        edit_count = 0

        # Post process the trajectories
        for json_file in json_files:
            with open(os.path.join(eval_folder, data_type, json_file), "r") as f:
                data = json.load(f)

            # Get the word to guess from the json file name
            word_to_guess = WordVariants.from_str(json_file.split("_")[0])

            t = 0
            done = False
            while t < len(data) and not done:
                action = data[t]["action"] if "action" in data[t] else data[t]["question"]

                # If the action leads to success
                if is_done(word_to_guess, action) and data[t]["reward"] == -1.0:
                    edit_dict["data_type"].append(data_type)
                    edit_dict["json_file"].append(json_file)
                    edit_dict["t"].append(t)
                    edit_dict["original_action"].append(action)
                    edit_dict["original_env_answer"].append(data[t]["answer"])

                    edit_count += 1

                    # Make a copy of the json file
                    copied_file_name = json_file.replace(".json", f"_original.json")
                    shutil.copy(os.path.join(eval_folder, data_type, json_file), os.path.join(eval_folder, data_type, copied_file_name))

                    # Edit the json file
                    data[t]["answer"] = "yes"
                    data[t]["reward"] = 0.0
                    data = data[:t+1]  # Since the agent actually succeeded at t, any rollout after t is invalid

                    # Write the edited json file
                    with open(os.path.join(eval_folder, data_type, json_file), "w") as f:
                        json.dump(data, f, indent=4)

                    done = True

                t += 1

        print(f"Edit count: {edit_count}")

    # Write the edit dict as a csv file
    if len(edit_dict["data_type"]) > 0:
        df = pd.DataFrame(edit_dict)
        df.to_csv(os.path.join(eval_folder, f"edit_dict_{curr_time_str}.csv"), index=False)

        