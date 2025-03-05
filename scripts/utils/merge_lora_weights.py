from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

mode = "critic"

if mode == "generator":
    # Paths
    base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
    lora_model_path = "/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/checkpoint-480"
    merged_model_path = "/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/merged_checkpoint-480"

    print(f"==== Mode: {mode} ====")
    print(f"Loading base model from {base_model_path}")
    print(f"Loading LoRA adapter from {lora_model_path}")
    print(f"Saving merged model to {merged_model_path}")
    input("Press Enter to continue...")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_model_path)

    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(merged_model_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)

    print(f"Merged model saved at: {merged_model_path}")
elif mode == "critic":
    # Paths
    base_model_path = "/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/merged_checkpoint-480"
    lora_model_path = "/share/portal/hw575/agent_prm/save/rm/250304_175736_iter1_-share-portal-hw575-agent_prm-save-sft-250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all-merged_checkpoint-480_peft=true_100-data/model"
    merged_model_path = "/share/portal/hw575/agent_prm/save/rm/250304_175736_iter1_-share-portal-hw575-agent_prm-save-sft-250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all-merged_checkpoint-480_peft=true_100-data/model/merged"

    print(f"==== Mode: {mode} ====")
    print(f"Loading base model from {base_model_path}")
    print(f"Loading LoRA adapter from {lora_model_path}")
    print(f"Saving merged model to {merged_model_path}")
    input("Press Enter to continue...")

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, torch_dtype=torch.float16, num_labels=1)

    # Load LoRA adapter (TODO: Now, it cannot get loaded properly)
    model = PeftModel.from_pretrained(model, lora_model_path)
    print(model)
    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(merged_model_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_model_path)
