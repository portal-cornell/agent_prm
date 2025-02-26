from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Paths
base_model_path = "meta-llama/Llama-3.2-3B-Instruct"
lora_model_path = "/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/checkpoint-480"
merged_model_path = "/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/merged_checkpoint-480"

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
