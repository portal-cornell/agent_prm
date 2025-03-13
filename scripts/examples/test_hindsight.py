import os
import json
import random
import copy
import pandas as pd
from tqdm import tqdm
from jinja2 import Template
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from agent_prm.utils.openai import generate_from_openai_completion
from agent_prm.utils.parser import parse_json
from agent_prm.envs.twenty_questions.data import get_default_word_list

NUM_RESPONSES_FOR_ALT_ACTION = 3
NUM_RESPONSES_FOR_PREF_ACTION = 3

save_folder_path = "/share/portal/hw575/agent_prm/playground/hindsight"
main_folder_path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/3B-PSFT-all-data-3epoches_250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/train"
files_to_test = [
    "Basketball_20.json",  # [Success] Repeated some of the same questions. Have some arbitrary questions
    "Whisk_20.json",  # [Failed] Ask specific objects too early. Object doesn't match previous descriptions
    "Dog_20.json", # [Failed] Suggesting some alternative options could have really helped.
    
    # "Scissors;Pair of scissors_20.json",  # [Optional] Ask some specific object that's not matching previous answer descriptions
    # "Sculpture_20.json",  # [Optional] Too hard, can't even narrow down the categories
]

# Map from file name to the answer
summary_dict = {
    "Basketball_20.json": "The agent began by determining that the secret word was not a living thing but a type of sports equipment, specifically a ball. The agent then narrowed the possibilities by confirming that the ball is used in a team sport typically played on a smaller court, dismissing options like Tennis and Volleyball. By the twelfth question, the agent successfully guessed that the word is 'Basketball.' The secret word is indeed 'basketball.'",
    "Whisk_20.json": "The agent began by determining that the secret word was not a living thing and ruled out categories like sports equipment, electronic devices, food, clothing, furniture, and office supplies. After discovering that the word was a type of tool, they specified it as a handheld tool. Despite narrowing it down to handheld tools, the agent guessed a series of tools commonly used for various functions including cutting, tightening, and loosening, but did not correctly identify the secret word. The agent unsuccessfully finished the 20 questions without guessing the word 'whisk.'",
    "Dog_20.json": "The agent began by narrowing down the secret word to a living thing and then further specified it as a mammal and a carnivore. Despite successfully identifying the general category, the agent struggled to guess the correct animal. The agent incorrectly guessed a variety of medium to small carnivorous mammals, such as weasel, fox, and raccoon, but never guessed 'dog,' which is the actual secret word. Additionally, the agent repeated 'Civet' in question 20, which they had already guessed in question 12. The agent did not succeed in guessing the secret word within the 20 questions.'",
}

all_obj_list = [wv[0] for wv in get_default_word_list("all")]


EXPERT_ERROR_TAGS = [
    "[fixate_summary]",
    "[lack_reason_about_options]",
    "[prefer_unlisted_words]",
    "[prefer_empty_question]",
]


def query_expert_preference(input_data, action, alt_action, prompt_template, num_responses=3):
    """
    Query the gpt expert with alternative questions and return the expert's preference

    Parameters:
    - input_data: dict
    - action: str
    - alt_action: str
    """
    system_prompt = prompt_template.render(system=True, **input_data)

    question_dict = {
        'action': {
            'question': action,
            'win_count': 0,
            'curr_win': 0
        },
        'alt_action': {
            'question': alt_action,
            'win_count': 0,
            'curr_win': 0
        }
    }

    question_order = ['action', 'alt_action']  # 0-th index corresponds to question A, 1-th index corresponds to question B

    total_cost = 0

    pref_response_list = []

    for _ in range(num_responses):
        # If one of the question is guaranteed to win (win_count > num_responses / 2), then we can stop querying the expert
        if question_dict['action']['win_count'] > num_responses / 2.0 or question_dict['alt_action']['win_count'] > num_responses / 2.0:
            print(json.dumps(question_dict, indent=4))
            print(f"One of the question is guaranteed to win, so we can stop querying the expert")
            continue

        random.shuffle(question_order)

        question_a = question_dict[question_order[0]]['question']
        question_b = question_dict[question_order[1]]['question']

        print(f"Question Order: {question_order}")
        print(f"Question A: {question_a}")
        print(f"Question B: {question_b}")

        input_prompt = prompt_template.render(system=False, mode="input", question_a=question_a, question_b=question_b, **input_data)
        print(input_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ]

        response, cost = generate_from_openai_completion(
            messages=messages, model="gpt-4o", temperature=0.7
        )
        total_cost += cost

        response_json = parse_json(response)
        try:
            assert response_json is not None, f"Failed to parse response: {response}"
            assert "reason" in response_json and "preferred_question" in response_json, f"Invalid response: {response}. Must contain 'reason' and 'preferred_question'"
            assert response_json['preferred_question'].lower() in ['a', 'b'], f"Invalid response: {response}. 'preferred_question'.lower() must be one of ['a', 'b']"
        except Exception as e:
            print(f"Error parsing response: {response}")
            raise e
        
        if response_json['preferred_question'].lower() == 'a':
            question_dict[question_order[0]]['win_count'] += 1
            question_dict[question_order[0]]['curr_win'] = 1
            response_json['preferred_question'] = question_order[0]
        elif response_json['preferred_question'].lower() == 'b':
            question_dict[question_order[1]]['win_count'] += 1
            question_dict[question_order[1]]['curr_win'] = 1
            response_json['preferred_question'] = question_order[1]
        else:
            print(f"Invalid response: {response}")
            win_idx = int(input("Please enter the index of the winning question [0=A/1=B]"))
            question_dict[question_order[win_idx]]['win_count'] += 1
            question_dict[question_order[win_idx]]['curr_win'] = 1
            response_json['preferred_question'] = question_order[win_idx]

        # The 'expert_pref_score' tracks whether the expert prefs the action over the alt_action
        #   0 --> alt action is preferred
        #   1 --> action is preferred
        response_json['expert_pref_score'] = question_dict['action']['curr_win']
            
        print(json.dumps(response_json, indent=4))
        print(json.dumps(question_dict, indent=4))
        # input("stop")

        # Clear the current win count
        question_dict['action']['curr_win'] = 0
        question_dict['alt_action']['curr_win'] = 0

        pref_response_list.append(response_json)

    # Determine overall which action has more wins
    if question_dict['action']['win_count'] > question_dict['alt_action']['win_count']:
        overall_pref_score = 1
    else:
        overall_pref_score = 0
    
    return pref_response_list, overall_pref_score, total_cost
        


def query_expert_alt_action(input_data, prompt_template, num_responses=3):
    """
    Query the gpt expert with alternative questions and return the expert's preference
    """
    system_prompt = prompt_template.render(system=True, num_responses=num_responses, **input_data)
    # print(system_prompt)

    input_prompt = prompt_template.render(system=False, mode="input", num_responses=num_responses, **input_data)
    print(input_prompt)
    # input("stop")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    total_cost = 0

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o", temperature=0.7
    )
    total_cost += cost

    reason_action_list = parse_json(response)
    try:
        assert reason_action_list is not None, f"Failed to parse response: {response}"
        for reason_action in reason_action_list:
            assert "reason" in reason_action and "question" in reason_action, f"Invalid response: {reason_action}. Must contain 'reason' and 'question'"
    except Exception as e:
        print(f"Error parsing response: {response}")
        raise e

    print(json.dumps(reason_action_list, indent=4))
    # input("stop")
    return reason_action_list, total_cost
    

def get_input_to_generate_summary():
    for file_name in files_to_test:
        path = os.path.join(main_folder_path, file_name)
        
        # Get the answer from the path name
        answer = file_name.split("_")[0].lower()

        with open(path, "r") as f:
            data = json.load(f)

        history_str = ""
        for i in range(len(data)):
            history_str += f"Question #{i+1}: {data[i]['action']}\nAnswer: {data[i]['answer']}\n"
        history_str += f"=======\nThe secret word is {answer}"

        with open(os.path.join(save_folder_path, 'history', file_name.split('.')[0] + "_history_str.txt"), "w") as f:
            f.write(history_str)


def get_chat_so_far(trajectory, t):
    chat_list = []

    for i in range(0, t):
        step = trajectory[i]
        chat_list.append({
            'question': step['action'],
            'answer': step['answer']
        })

    return chat_list


def human_main():
    for file_name in files_to_test:
        path = os.path.join(main_folder_path, file_name)

        if os.path.exists(os.path.join(save_folder_path, file_name)):
            print(f"Skipping {file_name} as it already exists")
            continue

        # Read the full history
        with open(os.path.join(save_folder_path, 'history', file_name.split('.')[0] + "_history_str.txt"), "r") as f:
            history_str = f.read()

        # Read the raw data
        with open(path, "r") as f:
            traj = json.load(f)

        for t in range(len(traj)):
            chat_list = get_chat_so_far(traj, t)
            print("======= Overall History =======")
            print(history_str)

            print(f"======= Chat so far at t={t} =======")
            for chat in chat_list:
                print(f"Question: {chat['question']} | Answer: {chat['answer']}")
            
            print("================")
            for alt in traj[t]['alternatives']:
                print(f"[A] Reasoning: {traj[t]['reason']}\n Question: {traj[t]['action']}")
                print(f"[B] Reasoning: {alt['reason']}\n Question: {alt['action']}")
                
                # Get the human preference:
                pref = input("Which one do you prefer? [A/B/N] ")
                if pref.lower() == "a":
                    pref_score = 1
                elif pref.lower() == "b":
                    pref_score = 0
                elif pref.lower() == "n":
                    # Either one is fine
                    pref_score = 0.5
                else:
                    pref_score = float(input("Please enter a valid preference [A/B/N]"))
                
                alt['human_pref_score'] = pref_score

        with open(os.path.join(save_folder_path, file_name), "w") as f:
            json.dump(traj, f, indent=4)

def expert_main(mode):
    assert mode in ["alt", "pref"], f"Invalid mode: {mode}, must be one of ['alt', 'pref']"
    
    with open("prompts/twenty_questions/twenty_question_expert_gen_prefered_action.j2", "r") as f:
        expert_alt_action_prompt = Template(f.read())

    with open("prompts/twenty_questions/twenty_question_expert_select_prefered_action.j2", "r") as f:
        expert_select_prefered_action_prompt = Template(f.read())

    cumulative_cost = 0

    for i in tqdm(range(len(files_to_test))):
        file_name = files_to_test[i]

        # Assume that we already got human labels 
        path = os.path.join(save_folder_path, file_name)

        with open(path, "r") as f:
            traj = json.load(f)

        summary = summary_dict[file_name]

        for t in range(len(traj)):
            chat_str = ""
            for i, chat in enumerate(get_chat_so_far(traj, t)):
                chat_str += f"Question #{i+1}: {chat['question']}\nAnswer #{i+1}: {chat['answer']}\n"
            chat_str = "Nothing has been asked yet." if chat_str == "" else chat_str

            input_data = {
                'all_obj_list': all_obj_list,
                'summary': summary,
                'observation_action_history': chat_str
            }

            if mode == "alt":
                if 'expert_alternatives' in traj[t]:
                    print(f"Skipping {file_name} at t={t} as it already has expert alternatives")
                    continue

                reason_action_list, cost = query_expert_alt_action(input_data, expert_alt_action_prompt, num_responses=NUM_RESPONSES_FOR_ALT_ACTION)
                cumulative_cost += cost

                # Update the trajectory with the expert's reasoning and action
                traj[t]['expert_alternatives'] = reason_action_list

                with open(os.path.join(save_folder_path, file_name), "w") as f:
                    json.dump(traj, f, indent=4)

            elif mode == "pref":
                action = traj[t]['action']

                for i, alt in enumerate(traj[t]['alternatives']):
                    if action == alt['action']:
                        print(f"Skipping {file_name} at t={t} as the alt action {i} = {alt['action']} is the same as the action {action}")
                        traj[t]['alternatives'][i]['expert_pref_score'] = 0.5
                        cost = 0
                    elif 'expert_pref_score' in alt:
                        print(f"Skipping {file_name} at t={t} as the alt action {i} = {alt['action']} already has an expert pref score")
                        cost = 0
                    else:
                        pref_response_list, overall_pref_score, cost = query_expert_preference(input_data, action, alt['action'], expert_select_prefered_action_prompt, num_responses=NUM_RESPONSES_FOR_PREF_ACTION)
                        cumulative_cost += cost

                        traj[t]['alternatives'][i]['expert_pref_score'] = overall_pref_score
                        traj[t]['alternatives'][i]['expert_pref_responses'] = pref_response_list

                    with open(os.path.join(save_folder_path, file_name), "w") as f:
                        json.dump(traj, f, indent=4)

            print(f"t={t} | cost={cost} | cumulative_cost={cumulative_cost}")
     

def consolidate_expert_pref_scores():
    """
    Find the points of disagreement between human and expert.

    The difference will be stored as .csv with the following columns
    - file_name
    - t
    - alt_action_idx
    - observation_action_history
    - action
    - alt_action
    """
    human_pref_1_expert_pref_0_dict = {
        'file_name': [],
        't': [],
        'alt_action_idx': [],
        'observation_action_history': [],
        'action': [],
        'alt_action': [],
        'inspection_comments': []
    }

    for i in range(NUM_RESPONSES_FOR_ALT_ACTION):
        human_pref_1_expert_pref_0_dict[f'expert_pref_reason_{i}'] = []
        human_pref_1_expert_pref_0_dict[f'expert_pref_score_{i}'] = []

    human_pref_0_expert_pref_1_dict = copy.deepcopy(human_pref_1_expert_pref_0_dict)

    human_pref_0_expert_pref_0_count = 0
    human_pref_1_expert_pref_1_count = 0

    for file_name in files_to_test:
        print(f"Processing {file_name}")
        path = os.path.join(save_folder_path, file_name)

        with open(path, "r") as f:
            traj = json.load(f)
        
        for t in range(len(traj)):
            chat_str = ""
            for i, chat in enumerate(get_chat_so_far(traj, t)):
                chat_str += f"Question #{i+1}: {chat['question']}\nAnswer #{i+1}: {chat['answer']}\n"
            chat_str = "Nothing has been asked yet." if chat_str == "" else chat_str

            action = traj[t]['action']

            for i, alt in enumerate(traj[t]['alternatives']):
                if alt['human_pref_score'] == 1 and alt['expert_pref_score'] == 0:
                    dict_to_add = human_pref_1_expert_pref_0_dict
                elif alt['human_pref_score'] == 0 and alt['expert_pref_score'] == 1:
                    dict_to_add = human_pref_0_expert_pref_1_dict
                elif alt['human_pref_score'] == 0 and alt['expert_pref_score'] == 0:
                    human_pref_0_expert_pref_0_count += 1
                    continue
                elif alt['human_pref_score'] == 1 and alt['expert_pref_score'] == 1:
                    human_pref_1_expert_pref_1_count += 1
                    continue
                else:
                    continue

                dict_to_add['file_name'].append(file_name)
                dict_to_add['t'].append(t)
                dict_to_add['alt_action_idx'].append(i)
                dict_to_add['observation_action_history'].append(chat_str)
                dict_to_add['action'].append(action)
                dict_to_add['alt_action'].append(alt['action'])
                dict_to_add['inspection_comments'].append('') # Placeholder for ad-hoc inspection comments

                for j in range(NUM_RESPONSES_FOR_ALT_ACTION):
                    dict_to_add[f'expert_pref_reason_{j}'].append(alt['expert_pref_responses'][j]['reason'] if j < len(alt['expert_pref_responses']) else "")
                    dict_to_add[f'expert_pref_score_{j}'].append(alt['expert_pref_responses'][j]['expert_pref_score'] if j < len(alt['expert_pref_responses']) else None)

    df_human_pref_1_expert_pref_0 = pd.DataFrame(human_pref_1_expert_pref_0_dict)
    df_human_pref_0_expert_pref_1 = pd.DataFrame(human_pref_0_expert_pref_1_dict)

    df_human_pref_1_expert_pref_0.to_csv(os.path.join(save_folder_path, "human_pref_1_expert_pref_0.csv"), index=False)
    df_human_pref_0_expert_pref_1.to_csv(os.path.join(save_folder_path, "human_pref_0_expert_pref_1.csv"), index=False)

    # Create the confusion matrix (y-axis: human pref, x-axis: expert pref)
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = human_pref_0_expert_pref_0_count
    confusion_matrix[0, 1] = len(human_pref_0_expert_pref_1_dict['file_name'])  # This will get updated after we manually inspect the points of disagreement
    confusion_matrix[1, 0] = len(human_pref_1_expert_pref_0_dict['file_name'])  # This will get updated after we manually inspect the points of disagreement
    confusion_matrix[1, 1] = human_pref_1_expert_pref_1_count

    # Save the confusion matrix as a csv and a visualization
    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=['human_pref_0', 'human_pref_1'], columns=['expert_pref_0', 'expert_pref_1'])
    df_confusion_matrix.to_csv(os.path.join(save_folder_path, "confusion_matrix.csv"), index=False)

    # Plot the confusion matrix
    plot_confusion_matrix(df_confusion_matrix)

def plot_confusion_matrix(df_confusion_matrix, file_name="confusion_matrix.png", is_percentage=False):
    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(df_confusion_matrix, cmap='Blues', interpolation='nearest')
    # Add text to the cells
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{df_confusion_matrix.iloc[i, j]:.2f}" if not is_percentage else f"{df_confusion_matrix.iloc[i, j]:.2f}%", ha='center', va='center', color='black', fontsize=20)
    plt.colorbar()
    plt.xticks(ticks=[0, 1], labels=['expert_pref_0', 'expert_pref_1'])
    plt.yticks(ticks=[0, 1], labels=['human_pref_0', 'human_pref_1'])
    plt.xlabel('Expert Preference')
    plt.ylabel('Human Preference')
    plt.title(file_name)
    plt.savefig(os.path.join(save_folder_path, file_name))


def gen_consolidated_expert_pref_scores():
    """
    Update the confusion matrix based on the ad-hoc inspection comments
    """
    df_confusion_matrix = pd.read_csv(os.path.join(save_folder_path, "confusion_matrix.csv"))
    df_human_pref_1_expert_pref_0 = pd.read_csv(os.path.join(save_folder_path, "human_pref_1_expert_pref_0.csv"))
    df_human_pref_0_expert_pref_1 = pd.read_csv(os.path.join(save_folder_path, "human_pref_0_expert_pref_1.csv"))

    # Count the number of points that doesn't have the label [human_mislabel] in the inspection_comments column
    df_human_pref_1_expert_pref_0['human_mislabel'] = df_human_pref_1_expert_pref_0['inspection_comments'].apply(lambda x: 1 if '[human_mislabel]' in x or '[either]' in x else 0)
    df_human_pref_0_expert_pref_1['human_mislabel'] = df_human_pref_0_expert_pref_1['inspection_comments'].apply(lambda x: 1 if '[human_mislabel]' in x or '[either]' in x else 0)

    # Update the confusion matrix
    df_confusion_matrix.iloc[0, 1] = df_human_pref_0_expert_pref_1['human_mislabel'].value_counts()[0]
    df_confusion_matrix.iloc[1, 0] = df_human_pref_1_expert_pref_0['human_mislabel'].value_counts()[0]

    df_confusion_matrix.to_csv(os.path.join(save_folder_path, "confusion_matrix.csv"), index=False)
    plot_confusion_matrix(df_confusion_matrix)

    # Also plot the confusion matrix by percentage
    total_count = df_confusion_matrix.sum().sum()
    df_confusion_matrix_percentage = df_confusion_matrix.div(total_count).mul(100)
    plot_confusion_matrix(df_confusion_matrix_percentage, file_name="confusion_matrix_percentage.png", is_percentage=True)


def count_expert_error_tags():
    """
    Count the number of times each expert error tag appears in the inspection_comments column
    """
    df_human_pref_1_expert_pref_0 = pd.read_csv(os.path.join(save_folder_path, "human_pref_1_expert_pref_0.csv"))
    df_human_pref_0_expert_pref_1 = pd.read_csv(os.path.join(save_folder_path, "human_pref_0_expert_pref_1.csv"))

    results_dict = {
        "human_pref_1_expert_pref_0": {},
        "human_pref_0_expert_pref_1": {},
        "overall": {}
    }
    for tag in EXPERT_ERROR_TAGS:
        results_dict["human_pref_1_expert_pref_0"][tag] = df_human_pref_1_expert_pref_0['inspection_comments'].apply(lambda x: 1 if tag in x else 0).sum()
        results_dict["human_pref_0_expert_pref_1"][tag] = df_human_pref_0_expert_pref_1['inspection_comments'].apply(lambda x: 1 if tag in x else 0).sum()
        results_dict["overall"][tag] = results_dict["human_pref_1_expert_pref_0"][tag] + results_dict["human_pref_0_expert_pref_1"][tag]
    
    total_count = sum(results_dict["overall"].values())

    # Normalize the results by the total count
    for key, value in results_dict.items():
        for tag, count in value.items():
            value[tag] = count / total_count * 100
    
    colors = matplotlib.color_sequences['Dark2']

    # Create a bar plot of the results (with 3 subplots: human_pref_1_expert_pref_0, human_pref_0_expert_pref_1, overall)
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    for i, (key, value) in enumerate(results_dict.items()):
        axs[i].bar(value.keys(), value.values(), color=colors)
        # Add the percentage on top of the bar
        for j, (tag, count) in enumerate(value.items()):
            axs[i].text(j, count, f"{count:.2f}%", ha='center', va='bottom', fontsize=18)
        axs[i].set_title(f"{key} (n={sum(results_dict[key].values()) / 100 * total_count:.2f})")
        axs[i].set_xlabel('Failure Mode')
        axs[i].set_ylabel('% (of all disagreements)')
        axs[i].set_ylim(0, 100)

    # Add a legend to the plot
    # Create legend patches matching bar colors
    patches = [mpatches.Patch(color=colors[i], label=tag) for i, tag in enumerate(EXPERT_ERROR_TAGS)]

    # Add a single legend for all plots
    fig.legend(handles=patches, loc='upper center', ncol=len(EXPERT_ERROR_TAGS), fontsize=20)

    plt.savefig(os.path.join(save_folder_path, "expert_error_tags.png"))

if __name__ == "__main__":
    # get_input_to_generate_summary()
    # main()
    # human_main()
    # expert_main("pref")
    consolidate_expert_pref_scores()
    gen_consolidated_expert_pref_scores()
    # count_expert_error_tags()