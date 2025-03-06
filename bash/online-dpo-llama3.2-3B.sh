#!/bin/bash
DOMAIN=twenty_questions # alfworld, twenty_questions
DATA_DIR=iter1

TRAIN_SPLITS=train_10k
TEST_SPLITS=val
TRAIN_EPOCHS=1

POLICY_MODEL="/share/portal/hw575/agent_prm/save/sft/250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all/merged_checkpoint-480"
REWARD_MODEL="/share/portal/hw575/agent_prm/save/rm/250303_235634_iter1_-share-portal-hw575-agent_prm-save-sft-250224_225421_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=true_epoch3+all-merged_checkpoint-480_peft=false_10k-data/checkpoint-1250"

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
SAVE_DIR=save/online_dpo/${current_date}_${DATA_DIR}_${POLICY_MODEL//\//-}${NOTE}/

echo "Save directory: $SAVE_DIR"

accelerate launch  --num-processes 5 \
    --config_file configs/ds_configs/deepspeed_zero2.yaml scripts/train/online_dpo_vllm_thread.py \
    --dataset_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_train_splits ${TRAIN_SPLITS} \
    --dataset_eval_mixer "{\"${DATASET}\": 1.0}" \
    --dataset_eval_splits ${TEST_SPLITS} \
    --domain_name ${DOMAIN} \
    --model_name_or_path ${POLICY_MODEL} \
    --reward_model_path ${REWARD_MODEL} \
    --non_stop_penalty \
    --stop_token eos \
    --learning_rate 8e-7 \
    --total_episodes 50000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 6 \
    --gradient_checkpointing True \
    --max_prompt_token_length 2000 \
    --response_length 256 \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --beta 0.03 \
    --temperature 0.7 \
    --num_generation_per_prompt 2 \
    --sanity_check_max_samples 128 \
    --output_dir ${SAVE_DIR} \
    --checkpoint_output_dir tmp/chkpts/ \
    --save_freq 50 \
    --vllm_device cuda:5 \
    --vllm_gpu_memory_utilization 0.9 \
    --hf_metadata_dataset "" \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    --wandb_project_name "LLM_RM" \
    --with_tracking
