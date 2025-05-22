#!/bin/bash
WANDB_PROJECT_NAME="Hinsight_LLM"
export WANDB_PROJECT=$WANDB_PROJECT_NAME

DATA_DIRS=""

LEARNING_RATE=3e-5
EVAL_STEPS=20

######################## Twenty Questions #########################################
DOMAIN=twenty_questions
MAX_SEQ_LENGTH=4000
TRAIN_BATCH_SIZE=4
GRAD_ACCUM_STEPS=16
GPU_COUNT=2


# ############################################## Car Dealer ##############################################
# DOMAIN=car_dealer
# MAX_SEQ_LENGTH=3500
# TRAIN_BATCH_SIZE=2
# GRAD_ACCUM_STEPS=12
# GPU_COUNT=4

############################################## Guess My City ##############################################
# DOMAIN=guess_my_city
# MAX_SEQ_LENGTH=3000
# TRAIN_BATCH_SIZE=4
# GRAD_ACCUM_STEPS=16
# EVAL_STEPS=16
# GPU_COUNT=2

MODEL="TODO"
DATA_DIR="TODO"
EPOCHS=TODO
SAVE_STEPS=TODO
MODEL_LOG_NAME="TODO"

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

export TRITON_CACHE_DIR=/REDACTED
TRITON_CACHE_DIR=/REDACTED/.triton

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
    --eval_steps $EVAL_STEPS \
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