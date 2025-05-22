#!/bin/bash

TRAIN_SPLITS=train_10k
TEST_SPLITS=val
TRAIN_EPOCHS=1

# Default values
TRAIN_BATCH_SIZE=2
GRAD_ACC=1 # When we use 4 GPUs (3 to train, 1 to generate responses)
NUM_PROCESSES=3 # When we use 4 GPUs
VLLM_DEVICE="cuda:3"
EXPLORATION_PROB=0.5

########################################################################################
# Checklist
# - Verify DATA_DIR
# - Verify MODEL
# - Verify MODEL_LOG_NAME
########################################################################################
DOMAIN=twenty_questions
SAVE_FREQ=167  # 10000 / 3 / 2 / 1 = 1666.6666666667 --> 1667/10=166.7 --> 167
MAX_SEQ_LENGTH=2000
OUTPUT_LENGTH=256

# # # #### Hindsight iter1 (with LEAP exploration model)
# DATA_DIR="iter1_hindsight-biased-on-60_with-summary"
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-80pct-lr=5e-6_hindsight-biased-on-60_with-summary
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250511_172545_iter1_hindsight-biased-on-60_with-summary_pi0_hindsight-biased-on-60_with-summary_peft=false/model/checkpoint-1000"
# # Hindsight model (LEAP pi1)
# HINDSIGHT_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250427_163704_iter1_leap_pi1_leap_from-base_epochs=1/checkpoint-16"
# MODEL_LOG_NAME="pi1_Q0-lr=5e-6-with-summary_hindsight-biased-on-60_with-summary_leap-pi1-exploration"
# HINDSIGHT_TYPE="leap" # "leap" or "counterfactual"

# #### Hindsight iter1 (with Hindsight Generator exploration model)
# DATA_DIR="iter1_hindsight-biased-on-60_with-summary"
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-80pct-lr=5e-6_hindsight-biased-on-60_with-summary
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250511_172545_iter1_hindsight-biased-on-60_with-summary_pi0_hindsight-biased-on-60_with-summary_peft=false/model/checkpoint-1000"
# # Hindsight model: first trained on all counterfactual data (1 epoch). then train on counterfactual data that leads to success (2 epochs)
# HINDSIGHT_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250512_014345_iter1_hindsight_pi1_hindsight_trained-from-hindsight-all_lr=3e-6_epochs=2/checkpoint-22"
# MODEL_LOG_NAME="pi1_Q0-lr=3e-6-with-summary_hindsight-biased-on-60_with-summary_hindsight-generator-exploration"
# HINDSIGHT_TYPE="counterfactual" # "leap" or "counterfactual"

# # # #### Vanilla iter1 (with Hindsight Generator exploration model)
# DATA_DIR="iter1_hindsight-biased-on-60_with-summary"
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250403_003704_iter1_pi0-new-env_pi0_pi0-new-env_peft=false_pi0-new-env/model/checkpoint-750"
# # Hindsight model: first trained on all counterfactual data (1 epoch). then train on counterfactual data that leads to success (2 epochs)
# HINDSIGHT_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250512_014345_iter1_hindsight_pi1_hindsight_trained-from-hindsight-all_lr=3e-6_epochs=2/checkpoint-22"
# MODEL_LOG_NAME="pi1_Q0-60pct-lr=5e-6_pi0-new-env_hindsight-generator-exploration_hindsight-biased-on-60_with-summary"
# HINDSIGHT_TYPE="counterfactual" # "leap" or "counterfactual"

# # #### Hindsight iter1 (with Hindsight Generator exploration model) - exploration prob = 0.75
# DATA_DIR="iter1_hindsight-biased-on-60_with-summary"
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-60
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250411_012114_iter1_hindsight-biased-on-60_pi0_hindsight-biased-on-60_peft=false/model/checkpoint-1250"
# # Hindsight model: first trained on all counterfactual data (1 epoch). then train on counterfactual data that leads to success (2 epochs)
# HINDSIGHT_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250512_014345_iter1_hindsight_pi1_hindsight_trained-from-hindsight-all_lr=3e-6_epochs=2/checkpoint-22"
# MODEL_LOG_NAME="pi1_Q0-lr=3e-6-vanilla_hindsight-biased-on-60_with-summary_hindsight-generator-exploration=75pct_redo"
# HINDSIGHT_TYPE="counterfactual" # "leap" or "counterfactual"
# EXPLORATION_PROB=0.75

# # # #### Vanilla iter1 (with Hindsight Generator exploration model) - exploration prob = 0.75
DATA_DIR="iter1_hindsight-biased-on-60_with-summary"
# pi0-all-data-3epoches
POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env
REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250403_003704_iter1_pi0-new-env_pi0_pi0-new-env_peft=false_pi0-new-env/model/checkpoint-750"
# Hindsight model: first trained on all counterfactual data (1 epoch). then train on counterfactual data that leads to success (2 epochs)
HINDSIGHT_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250512_014345_iter1_hindsight_pi1_hindsight_trained-from-hindsight-all_lr=3e-6_epochs=2/checkpoint-22"
MODEL_LOG_NAME="pi1_Q0-60pct-lr=5e-6_pi0-new-env_hindsight-generator-exploration=75pct_hindsight-biased-on-60_with-summary"
HINDSIGHT_TYPE="counterfactual" # "leap" or "counterfactual"
EXPLORATION_PROB=0.75

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
SAVE_DIR=save/${DOMAIN}/online_dpo/${current_date}_${DATA_DIR}_${MODEL_LOG_NAME}${NOTE}/

echo "Save directory: $SAVE_DIR"

accelerate launch  --num-processes ${NUM_PROCESSES} \
    --config_file configs/ds_configs/deepspeed_zero2.yaml scripts/train/online_dpo_vllm_exploration_with_summary_prompt.py \
    --dataset_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_eval_splits ${TEST_SPLITS} \
    --domain_name ${DOMAIN} \
    --model_name_or_path ${POLICY_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --hindsight_model_name_or_path ${HINDSIGHT_MODEL} \
    --hindsight_type ${HINDSIGHT_TYPE} \
    --exploration_prob ${EXPLORATION_PROB} \
    --non_stop_penalty \
    --penalty_reward_value -10.0 \
    --stop_token eos \
    --learning_rate 8e-8 \
    --total_episodes 10000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --gradient_checkpointing True \
    --max_prompt_token_length ${MAX_SEQ_LENGTH} \
    --response_length ${OUTPUT_LENGTH} \
    --min_response_length 1 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --beta 0.03 \
    --temperature 0.7 \
    --num_generation_per_prompt 2 \
    --sanity_check_max_samples 128 \
    --output_dir ${SAVE_DIR} \
    --checkpoint_output_dir tmp/chkpts/ \
    --save_freq ${SAVE_FREQ} \
    --vllm_device ${VLLM_DEVICE} \
    --vllm_gpu_memory_utilization 0.9 \
    --hf_metadata_dataset "" \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    --wandb_project_name "LLM_RM" \
    --with_tracking