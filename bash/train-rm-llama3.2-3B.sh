#!/bin/bash
# Default values
NUM_PROCESSES=2  # For 20 questions and guess my city, this is sufficient
NUM_EVALS=10

########################################################################################
# Checklist
# - Verify DATA_DIR
# - Verify MODEL
# - Verify MODEL_LOG_NAME
########################################################################################

DOMAIN=twenty_questions
SAVE_FREQ=250
MAX_SEQ_LEN=2048

#=====================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ CAR DEALER
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#=====================================================================================================
# DOMAIN=car_dealer
# SAVE_FREQ=500
# MAX_SEQ_LEN=3500

#=====================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ GUESS MY CITY
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#=====================================================================================================
# DOMAIN=guess_my_city
# SAVE_FREQ=250
# MAX_SEQ_LEN=3000

DATA_DIR="TODO"
MODEL="TODO"
MODEL_LOG_NAME="TODO"

###################################################################################################################################
TRAIN_SPLITS=train
TEST_SPLITS=val
TRAIN_EPOCHS=1

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

export TRITON_CACHE_DIR=/REDACTED
TRITON_CACHE_DIR=/REDACTED/.triton

DATASET=data/${DOMAIN}/prm/${DATA_DIR}
SAVE_DIR=save/${DOMAIN}/rm/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}_peft=${USE_PEFT}${NOTE}/model
EVAL_DIR=save/${DOMAIN}/rm/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}_peft=${USE_PEFT}${NOTE}/eval
echo "Save directory: $SAVE_DIR"
echo "Eval directory: $EVAL_DIR"


# Default learning rate was 5e-5

accelerate launch  --num-processes ${NUM_PROCESSES} \
    --config_file configs/ds_configs/deepspeed_zero3.yaml scripts/train/rm.py \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_splits ${TEST_SPLITS} \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --domain_name ${DOMAIN} \
    --learning_rate 5e-6 \
    --use_peft ${USE_PEFT} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --max_token_length ${MAX_SEQ_LEN} \
    --max_prompt_token_length ${MAX_SEQ_LEN} \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --num_evals ${NUM_EVALS} \
    --save_freq ${SAVE_FREQ} \
    --output_dir ${SAVE_DIR} \
    --eval_dir ${EVAL_DIR} \
    --gradient_checkpointing \
    --seed 2 \
    --with_tracking \
