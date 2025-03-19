#!/bin/bash

########################################################################################
# Checklist
# - Verify DATA_DIR
# - Verify MODEL
# - Verify MODEL_LOG_NAME
########################################################################################

DOMAIN=twenty_questions # alfworld, twenty_questions
DATA_DIR=iter3

TRAIN_SPLITS=train_10k
TEST_SPLITS=val
TRAIN_EPOCHS=1

############## iter1
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"

# 5e-5, redo: the best is 80 pct
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/rm/250311_155205_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_peft=false_lr=5e-5/model/checkpoint-1000"
# 5e-6, best is 80pct
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/rm/250311_155325_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_peft=false_lr=5e-6/model/checkpoint-1000"

############## iter2
# # pi1-80pct_Q0-80pct-lr=5e-6
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6/checkpoint-100"
# # 60pct (which has the best overall performance)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/rm/250314_210211_iter2_pi1-80pct_Q0-80pct-lr=5e-6_peft=false/model/checkpoint-750"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6"

############## iter3
# pi2-60pct_Q1-60pct-lr=5e-6
POLICY_MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250316_230359_iter2_pi2_Q1-60pct-lr=5e-6_Q1-60pct/checkpoint-75"
# BoN_pi2_Q2-80pct-lr=5e-6 (80pct)
REWARD_MODEL="/share/portal/hw575/agent_prm/save/rm/250318_100344_iter3_pi2-60pct_Q1-60pct-lr=5e-6_peft=false/model/checkpoint-1000"
MODEL_LOG_NAME="pi3_Q2-80pct-lr=5e-6"


current_date=$(date +"%y%m%d_%H%M%S")

# Default values
NOTE=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --note)
            NOTE="_$2"
            shift
            ;;
    esac
    shift
done

echo "Note: $NOTE"

export TRITON_CACHE_DIR=/share/portal/hw575
TRITON_CACHE_DIR=/share/portal/hw575/.triton

DATASET=data/${DOMAIN}/prm/${DATA_DIR}
SAVE_DIR=save/online_dpo/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}${NOTE}/

echo "Save directory: $SAVE_DIR"

accelerate launch  --num-processes 3 \
    --config_file configs/ds_configs/deepspeed_zero2.yaml scripts/train/online_dpo_vllm_thread.py \
    --dataset_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_eval_splits ${TEST_SPLITS} \
    --domain_name ${DOMAIN} \
    --model_name_or_path ${POLICY_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --non_stop_penalty \
    --penalty_reward_value -10.0 \
    --stop_token eos \
    --learning_rate 8e-7 \
    --total_episodes 10000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --max_prompt_token_length 2000 \
    --response_length 256 \
    --min_response_length 1 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --beta 0.03 \
    --temperature 0.7 \
    --num_generation_per_prompt 2 \
    --sanity_check_max_samples 128 \
    --output_dir ${SAVE_DIR} \
    --checkpoint_output_dir tmp/chkpts/ \
    --save_freq 40 \
    --vllm_device cuda:3 \
    --vllm_gpu_memory_utilization 0.9 \
    --hf_metadata_dataset "" \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    --wandb_project_name "LLM_RM" \
    --with_tracking
