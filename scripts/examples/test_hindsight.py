import json

path = "/share/portal/hw575/agent_prm/data/twenty_questions/eval/iter0/3B-PSFT-all-data-3epoches_250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/train/Basketball_1.json"

# Get the answer from the path name
answer = path.split("/")[-1].split("_")[0].lower()

with open(path, "r") as f:
    data = json.load(f)

history_str = ""
for i in range(len(data)):
    history_str += f"Question #{i+1}: {data[i]['action']}\nAnswer: {data[i]['answer']}\n"
history_str += f"=======\nThe secret word is {answer}"

with open("history_str.txt", "w") as f:
    f.write(history_str)


from agent_prm.envs.twenty_questions.data import get_default_word_list

all_obj_list = [wv[0] for wv in get_default_word_list("all")]

with open("all_obj_list.txt", "w") as f:
    f.write(str(all_obj_list))
