#!/bin/bash
WANDB_PROJECT_NAME="Hinsight_LLM"
export WANDB_PROJECT=$WANDB_PROJECT_NAME

DATA_DIRS=""

LEARNING_RATE=3e-5
######################## Twenty Questions #########################################
DOMAIN=twenty_questions
MAX_SEQ_LENGTH=4000
TRAIN_BATCH_SIZE=4
GRAD_ACCUM_STEPS=16
GPU_COUNT=2

###################
# Vanilla mode
###################
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter0-all
# EPOCHS=3

###################
# Multi-STaR
###################
##### Iter1
# Option 1: Train from base model
# 29 steps (5 ckpts) 29/5=5.8 (so save_steps=6)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter1_multi-star
# EPOCHS=1
# SAVE_STEPS=6
# MODEL_LOG_NAME="pi1_multi-star_from-base"

# # Option 2: Train from pi0
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# DATA_DIR=iter1_multi-star
# EPOCHS=1
# SAVE_STEPS=6
# MODEL_LOG_NAME="pi1_multi-star_from-pi0"

##### Iter2
# Option 1: Train from base model
# 3995 * 1 / 2 / 4 / 16 = 31.2109375
# 31.2109375 / 5 = 6.2421875 (save_steps=7)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter2_multi-star
# EPOCHS=1
# SAVE_STEPS=7
# MODEL_LOG_NAME="pi2_multi-star_from-base"

##### Iter3
# Option 1: Train from base model
# 3627 * 1 / 2 / 4 / 16 = 28.3359375 --> 29
# 29/5 = 5.8 (save_steps=6)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star
# EPOCHS=1
# SAVE_STEPS=6
# MODEL_LOG_NAME="pi3_multi-star_from-base"

# ##### Iter1 (10k datapoints)
# # Option 1: Train from base model
# # 10000 * 1 / 2 / 4 / 16 = 78.125
# # 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter1_multi-star_10k
# EPOCHS=1
# SAVE_STEPS=16
# MODEL_LOG_NAME="pi1_multi-star_from-base_10k-data"

# ##### Iter1 (10k datapoints) - lower learning rate
# # Option 1: Train from base model
# # 10000 * 1 / 2 / 4 / 16 = 78.125
# # 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter1_multi-star_10k
# EPOCHS=1
# SAVE_STEPS=16
# LEARNING_RATE=3e-6
# MODEL_LOG_NAME="pi1_multi-star_from-base_10k-data_lr=3e-6"


##### Iter2 (10k datapoints)
# Option 1: Train from base model
# 10000 * 1 / 2 / 4 / 16 = 78.125
# 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter2_multi-star_10k
# EPOCHS=1
# SAVE_STEPS=16
# MODEL_LOG_NAME="pi2_multi-star_from-base_10k-data"

##### Iter2 (10k datapoints) - with 50% past rollouts
# Option 1: Train from base model
# 10000 * 1 / 2 / 4 / 16 = 78.125
# 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter2_multi-star_10k_mix-50pct-past
# EPOCHS=1
# SAVE_STEPS=16
# LEARNING_RATE=3e-6
# MODEL_LOG_NAME="pi2_multi-star_from-base_10k-data_50pct-past_lr=3e-6"

# ##### Iter3 (10k datapoints)
# # Option 1: Train from base model
# # 10000 * 1 / 2 / 4 / 16 = 78.125
# # 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k
# EPOCHS=1
# SAVE_STEPS=16
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data"

# ##### Iter3 (10k datapoints) - Small learning rate
# # Option 1: Train from base model
# # 10000 * 1 / 2 / 4 / 16 = 78.125
# # 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k
# EPOCHS=1
# SAVE_STEPS=16
# LEARNING_RATE=3e-6
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data_lr=3e-6"

##### Iter3 (10k datapoints) - Small gradient accumulation steps
# Option 1: Train from base model
# 10000 * 1 / 2 / 4 / 8 = 156.25
# 156 / 5 = 31.2 (save_steps=32)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k
# EPOCHS=1
# SAVE_STEPS=32
# GRAD_ACCUM_STEPS=8
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data_small-grad-accum=8"

##### Iter3 (10k datapoints) - with 50% past rollouts
# Option 1: Train from base model
# 10000 * 1 / 2 / 4 / 16 = 78.125
# 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k_mix-50pct-past
# EPOCHS=1
# SAVE_STEPS=16
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data_50pct-past"

##### Iter3 (10k datapoints) - with 50% past rollouts
# Option 1: Train from base model
# 10000 * 1 / 2 / 4 / 16 = 78.125
# 79 / 5 = 15.8 (save_steps=16)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k_mix-50pct-past
# EPOCHS=1
# SAVE_STEPS=16
# LEARNING_RATE=3e-6
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data_50pct-past_lr=3e-6"

# ##### Iter3 (10k datapoints) - 70% past rollouts
# # Option 1: Train from base model
# # 10000 * 1 / 2 / 2 / 16 = 156.25
# # 156 / 5 = 31.2 (save_steps=32)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k_mix-70pct-past
# EPOCHS=1
# SAVE_STEPS=32
# TRAIN_BATCH_SIZE=2
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data_70pct-past"

##### Iter3 (10k datapoints) - 90% past rollouts
# Option 1: Train from base model
# 10000 * 1 / 2 / 2 / 16 = 156.25
# 156 / 5 = 31.2 (save_steps=32)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_multi-star_10k_mix-90pct-past
# EPOCHS=1
# SAVE_STEPS=32
# TRAIN_BATCH_SIZE=2
# MODEL_LOG_NAME="pi3_multi-star_from-base_10k-data_90pct-past"

###################
# LEAP
###################
##### Iter1
# Option 1: Train from base model
# 36 steps (5 ckpts) 36/5=7.2 (so save_steps=8)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter1_leap
# EPOCHS=1
# SAVE_STEPS=8
# MODEL_LOG_NAME="pi1_leap_from-base"

# # Option 2: Train from pi0
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# DATA_DIR=iter1_leap
# EPOCHS=1
# SAVE_STEPS=8
# MODEL_LOG_NAME="pi1_leap_from-pi0"

# ##### Iter1 (lower learning rate)
# # Option 1: Train from base model
# # 36 steps (5 ckpts) 36/5=7.2 (so save_steps=8)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter1_leap
# EPOCHS=1
# SAVE_STEPS=8
# LEARNING_RATE=3e-6
# MODEL_LOG_NAME="pi1_leap_from-base_lr=3e-6"

##### Iter2
# Option 1: Train from base model
# 4861 * 1 / 2 / 4 / 16 = 37.9765625
# 37.9765625 / 5 = 7.5953125 (save_steps=8)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter2_leap
# EPOCHS=1
# SAVE_STEPS=8
# MODEL_LOG_NAME="pi2_leap_from-base"

##### Iter2 (lower learning rate)
# Option 1: Train from base model
# 4861 * 1 / 2 / 4 / 16 = 37.9765625
# 37.9765625 / 5 = 7.5953125 (save_steps=8)
MODEL=meta-llama/Llama-3.2-3B-Instruct
DATA_DIR=iter2_leap
EPOCHS=1
SAVE_STEPS=8
LEARNING_RATE=3e-6
MODEL_LOG_NAME="pi2_leap_from-base_lr=3e-6"

##### Iter3
# Option 1: Train from base model
# 4781 * 1 / 2 / 4 / 16 = 37.3515625 --> 38
# 38 / 5 = 7.6 (save_steps=8)
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter3_leap
# EPOCHS=1
# SAVE_STEPS=8
# MODEL_LOG_NAME="pi3_leap_from-base"

######################## Car Dealer #########################################
# DOMAIN=car_dealer  # car_dealer, twenty_questions
# MAX_SEQ_LENGTH=6500
# TRAIN_BATCH_SIZE=2
# GRAD_ACCUM_STEPS=12
# GPU_COUNT=4

# ###################
# # Vanilla mode
# ###################
# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter0_api
# EPOCHS=3
# # num datapoints * epoch / gpu / batch size / gradient accumulation steps
# # 2429 * 3 / 2 / 2 / 10 = 182.175
# # 183 / 5 = 36.6 (save_steps=37)
# SAVE_STEPS=37
# MODEL_LOG_NAME="pi0_vanilla_api"

# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter0_response
# EPOCHS=3
# # num datapoints * epoch / gpu / batch size / gradient accumulation steps
# # 2429 * 3 / 2 / 2 / 10 = 182.175
# # 183 / 5 = 36.6 (save_steps=37)
# SAVE_STEPS=37
# MODEL_LOG_NAME="pi0_vanilla_response"

# MODEL=meta-llama/Llama-3.2-3B-Instruct
# DATA_DIR=iter0
# EPOCHS=3
# # num datapoints * epoch / gpu / batch size / gradient accumulation steps
# # 4858 * 3 / 4 / 2 / 12 = 151.8125
# # 152 / 5 = 30.4
# SAVE_STEPS=31
# MODEL_LOG_NAME="pi0_vanilla"


########################################################################################################################################

DATA_DIRS+="data/${DOMAIN}/sft/${DATA_DIR}"

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

SAVE_DIR=save/${DOMAIN}/sft/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}_epochs=${EPOCHS}${NOTE}/
echo "Save directory: $SAVE_DIR"

accelerate launch \
    --num_processes $GPU_COUNT \
    --config_file configs/ds_configs/deepspeed_zero3.yaml scripts/train/sft_trl.py \
    --data_dirs "${DATA_DIRS}" \
    --output_dir ${SAVE_DIR} \
    --model_name_or_path ${MODEL} \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs $EPOCHS \
    --gradient_checkpointing True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --packing False \
    --torch_dtype bfloat16 \
    --optim adamw_torch_fused \
    --learning_rate $LEARNING_RATE \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
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