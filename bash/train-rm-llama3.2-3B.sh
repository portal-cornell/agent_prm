#!/bin/bash

########################################################################################
# Checklist
# - Verify DATA_DIR
# - Verify MODEL
# - Verify MODEL_LOG_NAME
########################################################################################

DOMAIN=twenty_questions # alfworld, twenty_questions
# DATA_DIR=iter1_hindsight-biased-on-70 # add '_no-reason' if you want to train on the no-reason dataset

################ iter1
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0"
################ iter2
# MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6/checkpoint-100"
# MODEL_LOG_NAME="pi1-80pct_Q0-80pct-lr=5e-6"
################ iter3
# MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250316_230359_iter2_pi2_Q1-60pct-lr=5e-6_Q1-60pct/checkpoint-75"
# MODEL_LOG_NAME="pi2-60pct_Q1-60pct-lr=5e-6"

################ iter1 (HINDSIGHT INVESTIGATION)
##### with hindsight
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_rollout-hindsight"
##### (debug: pi0 and pi2 mix)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_pi0-pi2-mix"
# ##### (debug: pi0 rollout on new env)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_pi0-new-env"
# ##### with hindsight (REDO)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-redo"
##### with hindsight (50-50 rollout from on-policy and off-policy/hindsight)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-50-50"
# ##### with hindsight (biased filtering)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased"
# ##### with hindsight (biased filtering - 30-70 onpolicy/offpolicy)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-30"
# ##### with hindsight (biased filtering - 50-50 onpolicy/offpolicy)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-50"
##### with hindsight (biased filtering - 70-30 onpolicy/offpolicy)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-70"
# ##### with hindsight (biased filtering - 40-60 onpolicy/offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-40
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-40"
# #### with hindsight (biased filtering - 60-40 onpolicy/offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-60
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-60"
#### with hindsight (biased filtering - 60-40 onpolicy/offpolicy) - TRAIN FROM PRETRAINED LLAMA3.2-3B
# DATA_DIR=iter1_hindsight-biased-on-60
# MODEL="meta-llama/Llama-3.2-3B-Instruct"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-60_from-pretrained-3B"


################ iter2 (HINDSIGHT INVESTIGATION)
# ##### baseline
# DATA_DIR=iter2_new-env
# # pi1_Q0-60pct-lr=5e-6_pi0-new-env (best val model)
# MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250418_164049_iter1_pi0-new-env_pi1_Q0-60pct-lr=5e-6_pi0-new-env"
# MODEL_LOG_NAME="pi1_Q0-60pct-lr=5e-6_pi0-new-env"
# ##### hindsight: pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (DATA = 60% onpolicy, 40% offpolicy)
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (best val model)
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250419_134330_iter1_hindsight-biased-on-60_pi1_Q0-lr=5e-6_hindsight-biased-on-60/checkpoint-165"
# MODEL_LOG_NAME="pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60"
# ##### hindsight: pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (DATA = 60% onpolicy, 40% offpolicy) FROM pi0 BASE MODEL
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi0-all-data-3epoches
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60"
# ##### hindsight: pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-40 (DATA = 40% onpolicy, 60% offpolicy)
# DATA_DIR=iter2_hindsight-biased-on-40
# # pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (best val model), still using the same model as the best val model
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250419_134330_iter1_hindsight-biased-on-60_pi1_Q0-lr=5e-6_hindsight-biased-on-60/checkpoint-165"
# MODEL_LOG_NAME="pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60"
##### hindsight: pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-40 (DATA = 50% onpolicy, 50% offpolicy)
# DATA_DIR=iter2_hindsight-biased-on-50
# pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (best val model), still using the same model as the best val model
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250419_134330_iter1_hindsight-biased-on-60_pi1_Q0-lr=5e-6_hindsight-biased-on-60/checkpoint-165"
# MODEL_LOG_NAME="pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60"

################ iter3 (HINDSIGHT INVETIGATION)
# ###### baseline
# DATA_DIR=iter3_new-env
# # pi2-40pct_Q1-40pct-lr=5e-6_new-env
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250422_005316_iter2_new-env_pi2_Q1-40pct-lr=5e-6_new-env/checkpoint-110"
# MODEL_LOG_NAME="pi2-40pct_Q1-40pct-lr=5e-6_new-env"
# ###### baseline (pi2 from pi0)
# DATA_DIR=iter3_new-env_from-pi0
# # pi2_Q1-40pct-lr=5e-6_new-env_from-pi0
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250424_141029_iter2_new-env_pi2_Q1-40pct-lr=5e-6_new-env_from-pi0"
# MODEL_LOG_NAME="pi2_Q1-40pct-lr=5e-6_new-env_from-pi0"
# #### hindsight: pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0 (DATA = 60% onpolicy, 40% offpolicy)
# DATA_DIR=iter3_hindsight-biased-on-60
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250423_205810_iter2_hindsight-biased-on-60_pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0"
# #### hindsight: pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0 (DATA = 60% onpolicy, 40% offpolicy)
# TRYING TO ONLY HAVE ONPOLICY FAILURE
# DATA_DIR=iter3_hindsight-biased-on-60_with-past-fail
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250423_205810_iter2_hindsight-biased-on-60_pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60-only-onpolicy-failure_from-pi0"


################ iter1 other exploration baselines
# ######### with pi*
# DATA_DIR=iter1_best-pi-on-60
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_best-pi-on-60"
######### with explorative pi0
DATA_DIR=iter1_explorative-pi-on-60
MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
MODEL_LOG_NAME="pi0_explorative-pi-on-60"

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

export TRITON_CACHE_DIR=/share/portal/hw575
TRITON_CACHE_DIR=/share/portal/hw575/.triton

DATASET=data/${DOMAIN}/prm/${DATA_DIR}
SAVE_DIR=save/${DOMAIN}/rm/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}_peft=${USE_PEFT}${NOTE}/model
EVAL_DIR=save/${DOMAIN}/rm/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}_peft=${USE_PEFT}${NOTE}/eval
echo "Save directory: $SAVE_DIR"
echo "Eval directory: $EVAL_DIR"


# Default learning rate was 5e-5

accelerate launch  --num-processes 2 \
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
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --num_evals 20 \
    --save_freq 250 \
    --output_dir ${SAVE_DIR} \
    --eval_dir ${EVAL_DIR} \
    --gradient_checkpointing \
    --seed 2 \
    --with_tracking \
