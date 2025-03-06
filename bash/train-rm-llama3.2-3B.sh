#!/bin/bash

DOMAIN=twenty_questions # alfworld, twenty_questions
DATA_DIR=iter1

MODEL="/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/merged_checkpoint-480"

TRAIN_SPLITS=train
TEST_SPLITS=val
TRAIN_EPOCHS=2

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
echo "Note: $NOTE"

export TRITON_CACHE_DIR=/share/portal/hw575
TRITON_CACHE_DIR=/share/portal/hw575/.triton

DATASET=data/${DOMAIN}/prm/${DATA_DIR}
SAVE_DIR=save/rm/${current_date}_${DATA_DIR}_${MODEL//\//-}_peft=${USE_PEFT}${NOTE}/model
EVAL_DIR=save/rm/${current_date}_${DATA_DIR}_${MODEL//\//-}_peft=${USE_PEFT}${NOTE}/eval
echo "Save directory: $SAVE_DIR"
echo "Eval directory: $EVAL_DIR"

accelerate launch  --num-processes 2 \
    --config_file configs/ds_configs/deepspeed_zero3.yaml scripts/train/rm.py \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_splits ${TEST_SPLITS} \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --domain_name ${DOMAIN} \
    --learning_rate 5e-5 \
    --use_peft ${USE_PEFT} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --num_evals 20 \
    --save_freq 500 \
    --output_dir ${SAVE_DIR} \
    --eval_dir ${EVAL_DIR} \
    --gradient_checkpointing \
    --seed 2 \
    --with_tracking \
