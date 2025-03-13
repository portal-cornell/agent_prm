#!/bin/bash
WANDB_PROJECT_NAME="Hinsight_LLM"
export WANDB_PROJECT=$WANDB_PROJECT_NAME

DATA_DIR=iter0-all
MODEL=meta-llama/Llama-3.2-3B-Instruct
DATA_DIRS=""
DATA_DIRS+="data/twenty_questions/sft/${DATA_DIR}"

# Remove the trailing comma
DATA_DIRS=${DATA_DIRS%,}

current_date=$(date +"%y%m%d_%H%M%S")

# Default values
USE_PEFT=false
NOTE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_peft)
            USE_PEFT=true
            ;;
        --note)
            NOTE="_$2"
            shift
            ;;
    esac
    shift
done

echo "Use PEFT: $USE_PEFT"

export TRITON_CACHE_DIR=/share/portal/hw575
TRITON_CACHE_DIR=/share/portal/hw575/.triton

SAVE_DIR=save/sft/${current_date}_${DATA_DIR}_${MODEL//\//-}_peft=${USE_PEFT}${NOTE}/
echo "Save directory: $SAVE_DIR"

accelerate launch \
    --num_processes 2 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml scripts/train/sft_trl.py \
    --data_dirs "${DATA_DIRS}" \
    --output_dir ${SAVE_DIR} \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --gradient_checkpointing True \
    --max_seq_length 4000 \
    --packing False \
    --torch_dtype bfloat16 \
    --optim adamw_torch_fused \
    --learning_rate 3e-5 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --save_strategy steps \
    --save_steps 30 \
    --save_total_limit 5 \
    --load_best_model_at_end False \
    --metric_for_best_model eval_loss \
    --use_peft $USE_PEFT \
    --lora_alpha 64 \
    --lora_r 128 \
    --lora_dropout 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.3 \
    --warmup_steps 10 \
    --bf16 \
    --seed 42 \
    --report_to wandb \
    --wandb_project_name "${WANDB_PROJECT_NAME}" \
    --logging_first_step \
    --logging_steps 10 \