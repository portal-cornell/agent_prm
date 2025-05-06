#!/bin/bash

TRAIN_SPLITS=train_10k
TEST_SPLITS=val
TRAIN_EPOCHS=1

# Default values
GRAD_ACC=6 # When we use 4 GPUs (3 to train, 1 to generate responses)

########################################################################################
# Checklist
# - Verify DATA_DIR
# - Verify MODEL
# - Verify MODEL_LOG_NAME
########################################################################################

# DOMAIN=twenty_questions
# SAVE_FREQ=55
# MAX_SEQ_LENGTH=2000
# OUTPUT_LENGTH=256

############## iter1
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"

# 5e-5, redo: the best is 80 pct
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250311_155205_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_peft=false_lr=5e-5/model/checkpoint-1000"
# 5e-6, best is 80pct
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250311_155325_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_peft=false_lr=5e-6/model/checkpoint-1000"

############## iter2
# # pi1-80pct_Q0-80pct-lr=5e-6
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250312_091827_iter1_-share-portal-hw575-agent_prm-save-sft-250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all-checkpoint-120_pen=-10_Q0-80pct-lr=5e-6/checkpoint-100"
# # 60pct (which has the best overall performance)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250314_210211_iter2_pi1-80pct_Q0-80pct-lr=5e-6_peft=false/model/checkpoint-750"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6"

############## iter3
# # pi2-60pct_Q1-60pct-lr=5e-6
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/online_dpo/250316_230359_iter2_pi2_Q1-60pct-lr=5e-6_Q1-60pct/checkpoint-75"
# # BoN_pi2_Q2-80pct-lr=5e-6 (80pct)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250318_100344_iter3_pi2-60pct_Q1-60pct-lr=5e-6_peft=false/model/checkpoint-1000"
# MODEL_LOG_NAME="pi3_Q2-80pct-lr=5e-6"

############################################################################################################################################
#                          Hindsight Data
############################################################################################################################################
############## iter1 (with PRM trained with hindsight data)
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-80pct-lr=5e-6_hindsight
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250321_135855_iter1_hindsight_pi0_rollout-hindsight_peft=false/model/checkpoint-1000"
# MODEL_LOG_NAME="pi1_Q0-80pct-lr=5e-6-hindsight"

############## iter1 (with PRM trained with pi0 and pi2 mixed data)
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-60pct-lr=5e-6_pi0-pi2-mix
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250331_225610_iter1_pi0-pi2-mix_pi0_pi0-pi2-mix_peft=false/model/checkpoint-750"
# MODEL_LOG_NAME="1-60pct-lr=5e-6_pi0-pi2-mix"

############## iter1 (with PRM trained with pi0 rolling out on the new env)
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250403_003704_iter1_pi0-new-env_pi0_pi0-new-env_peft=false_pi0-new-env/model/checkpoint-750"
# MODEL_LOG_NAME="pi1_Q0-60pct-lr=5e-6_pi0-new-env"

# ## Because 20pct worked well for hindsight-redo, we use it for pi0-new-env
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250403_003704_iter1_pi0-new-env_pi0_pi0-new-env_peft=false_pi0-new-env/model/checkpoint-250"
# MODEL_LOG_NAME="pi1_Q0-20pct-lr=5e-6_pi0-new-env"

## Hindsight redo (using filtered data onpolicy,offpolicy)
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-lr=5e-6_hindsight-biased
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250406_152011_iter1_hindsight-biased_pi0_hindsight-biased_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi1_Q0-lr=5e-6_hindsight-biased"

# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-lr=5e-6_hindsight-biased
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250406_152011_iter1_hindsight-biased_pi0_hindsight-biased_peft=false/model/checkpoint-250"
# MODEL_LOG_NAME="pi1_Q0-20pct-lr=5e-6_hindsight-biased"

# pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased-on-50
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250409_011522_iter1_hindsight-biased-on-50_pi0_hindsight-biased-on-50_peft=false/model/checkpoint-250"
# MODEL_LOG_NAME="pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-50"

# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased-on-60
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250411_012114_iter1_hindsight-biased-on-60_pi0_hindsight-biased-on-60_peft=false/model/checkpoint-250"
# MODEL_LOG_NAME="pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-60"

############## iter1 (with PRM trained with the new hindsight data)
# # 100% on-policy pi0 data, 0% off-policy pi0 data
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env (the best model on the validation set)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250403_003704_iter1_pi0-new-env_pi0_pi0-new-env_peft=false_pi0-new-env/model/checkpoint-750"
# MODEL_LOG_NAME="pi1_Q0-60pct-lr=5e-6_pi0-new-env"

# # [The best data mix based on validation performance] 40% on-policy pi0 data, 60% off-policy pi0 data 
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-80pct-lr=5e-6_hindsight-biased-on-40 (the best model on the validation set)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250411_011907_iter1_hindsight-biased-on-40_pi0_hindsight-biased-on-40_peft=false/model/checkpoint-1000"
# MODEL_LOG_NAME="pi1_Q0-80pct-lr=5e-6_hindsight-biased-on-40"

# TODO: [The 2nd best data mix based on validation performance] 60% on-policy pi0 data, 40% off-policy pi0 data 
# pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-60 (the best model on the validation set)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250411_012114_iter1_hindsight-biased-on-60_pi0_hindsight-biased-on-60_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi1_Q0-lr=5e-6_hindsight-biased-on-60"

############## iter1 (with PRM trained with the new hindsight data)
# ###### 100% on-policy pi0, 0% off-policy pi0 data
# DATA_DIR=iter2_new-env
# # pi1_Q0-60pct-lr=5e-6_pi0-new-env
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250418_164049_iter1_pi0-new-env_pi1_Q0-60pct-lr=5e-6_pi0-new-env"
# # BoN_pi1_Q1-40pct-lr=5e-6_new-env
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250421_123000_iter2_new-env_pi1_Q0-60pct-lr=5e-6_pi0-new-env_peft=false/model/checkpoint-500"
# MODEL_LOG_NAME="pi2_Q1-40pct-lr=5e-6_new-env"
# ###### 100% on-policy pi0, 0% off-policy pi0 data (TRAIN POLICY FROM pi0)
# DATA_DIR=iter2_new-env
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi1_Q1-40pct-lr=5e-6_new-env
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250421_123000_iter2_new-env_pi1_Q0-60pct-lr=5e-6_pi0-new-env_peft=false/model/checkpoint-500"
# MODEL_LOG_NAME="pi2_Q1-40pct-lr=5e-6_new-env"
# ##### 60% on-policy pi0, 40% off-policy pi0 data
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (best val model)
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250419_134330_iter1_hindsight-biased-on-60_pi1_Q0-lr=5e-6_hindsight-biased-on-60/checkpoint-165"
# # BoN_pi1_Q1-60pct-lr=5e-6_hindsight-biased-on-60
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250421_161837_iter2_hindsight-biased-on-60_pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60_peft=false/model/checkpoint-750"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60"
# ##### 60% on-policy pi0, 40% off-policy pi0 data (TRAIN POLICY FROM pi0)
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi1_Q1-60pct-lr=5e-6_hindsight-biased-on-60
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250421_161837_iter2_hindsight-biased-on-60_pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60_peft=false/model/checkpoint-750"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60"
# ##### 60% on-policy pi0, 40% off-policy pi0 data (We used the wrong Q1 model!!!!!!!, should have been: BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60)
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60 (best val model)
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250419_134330_iter1_hindsight-biased-on-60_pi1_Q0-lr=5e-6_hindsight-biased-on-60/checkpoint-165"
# # BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60 (best val model)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250421_161837_iter2_hindsight-biased-on-60_pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi2_Q1-lr=5e-6_hindsight-biased-on-60"
# ##### 60% on-policy pi0, 40% off-policy pi0 data (We used the wrong Q1 model!!!!!!!, should have been: BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60) (TRAIN POLICY FROM pi0)
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60 (best val model)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250421_161837_iter2_hindsight-biased-on-60_pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi2_Q1-lr=5e-6_hindsight-biased-on-60"
# ##### 60% on-policy pi0, 40% off-policy pi0 data (Unfortunately, the correct Q1 model does not work well. However, we have a Q1 trained from pi0, so let's try it: BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60_from-pi0) (TRAIN POLICY FROM pi0)
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60_from-pi0 (best val model)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250423_205911_iter2_hindsight-biased-on-60_pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60_peft=false_from-pi0/model/checkpoint-1250"
# MODEL_LOG_NAME="pi2_Q1-lr=5e-6-from-pi0_hindsight-biased-on-60"

############## iter2 (with PRM trained with the new hindsight data)
# ###### 100% on-policy pi0, 0% off-policy pi0 data
# DATA_DIR=iter3_new-env
# # pi2-40pct_Q1-40pct-lr=5e-6_new-env
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/online_dpo/250422_005316_iter2_new-env_pi2_Q1-40pct-lr=5e-6_new-env/checkpoint-110"
# # BoN_pi2_Q2-80pct-lr=5e-6_new-env
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250423_023048_iter3_new-env_pi2-40pct_Q1-40pct-lr=5e-6_new-env_peft=false/model/checkpoint-1000"
# MODEL_LOG_NAME="pi3_Q2-80pct-lr=5e-6_new-env"
# ###### 100% on-policy pi0, 0% off-policy pi0 data (TRAIN POLICY FROM pi0)
# DATA_DIR=iter3_new-env_from-pi0
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi2_Q2-lr=5e-6_new-env-pi2-from-pi0
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250425_012544_iter3_new-env_from-pi0_pi2_Q1-40pct-lr=5e-6_new-env_from-pi0_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi3_Q2-lr=5e-6_new-env-pi2-from-pi0"
# # ##### 60% on-policy pi0, 40% off-policy pi0 data (TRAIN POLICY FROM pi0)
# DATA_DIR=iter3_hindsight-biased-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi2_Q2-lr=5e-6_hindsight-biased-on-60  (best val model)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250428_224143_iter3_hindsight-biased-on-60_pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi3_Q2-lr=5e-6_hindsight-biased-on-60"
# ##### 60% on-policy pi0, 40% off-policy pi0 data (TRAIN POLICY FROM pi0 - 100% onpolicy low q data
# DATA_DIR=iter3_hindsight-biased-on-60_with-past-fail
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi2_Q2-40pct-lr=5e-6_hindsight-biased-on-60-with-past-fail  (best val model)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250429_173031_iter3_hindsight-biased-on-60_with-past-fail_pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60-only-onpolicy-failure_from-pi0_peft=false/model/checkpoint-500"
# MODEL_LOG_NAME="pi3_Q2-40pct-lr=5e-6_hindsight-biased-on-60-with-past-fail"

############################################################################################################################################
#                          Alternative Exploration
############################################################################################################################################
# ###### Using the best pi (pi*) to explore
# DATA_DIR=iter1_best-pi-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-lr=5e-6_best-pi-on-60 (the best model on the validation set)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250501_112539_iter1_best-pi-on-60_pi0_best-pi-on-60_peft=false/model/checkpoint-1250"
# MODEL_LOG_NAME="pi1_Q0-lr=5e-6_best-pi-on-60"
###### Using the best pi (pi*) to explore
# DATA_DIR=iter1_explorative-pi-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-80pct-lr=5e-6_explorative-pi-on-60 (the best model on the validation set)
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250501_112654_iter1_explorative-pi-on-60_pi0_explorative-pi-on-60_peft=false/model/checkpoint-1000"
# MODEL_LOG_NAME="pi1_Q0-80pct-lr=5e-6_explorative-pi-on-60"
###### Using the high temp pi0 to explore
# DATA_DIR=iter1_high-temp-on-60
# # pi0-all-data-3epoches
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# # BoN_pi0_Q0-20pct-lr=5e-6_high-temp-pi-on-60
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/rm/250503_175239_iter1_high-temp-on-60_pi0_high-temp-pi-on-60_peft=false/model/checkpoint-250"
# MODEL_LOG_NAME="pi1_Q0-20pct-lr=5e-6_high-temp-pi-on-60"

############################################################################################################################################
#                          Car Dealer
############################################################################################################################################
DOMAIN=car_dealer

###################### Vanilla RL
######### Iter 1
# DATA_DIR=iter1
# # pi0-83pct
# POLICY_MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250430_180817_iter0_pi0_vanilla_epochs=3/checkpoint-124"
# # BoN_pi0_Q0-lr=5e-6
# REWARD_MODEL="/share/portal/hw575/agent_prm/save/car_dealer/rm/250502_055014_iter1_pi0-83pct_peft=false/model/checkpoint-2500"
# MODEL_LOG_NAME="pi1_Q0-lr=5e-6"
# # Manullay changing it
# # 20000 / 5 / 2 / 2 = 1000
# # 1000 / 5 = 200
# SAVE_FREQ=200
# MAX_SEQ_LENGTH=3500
# OUTPUT_LENGTH=400
# TRAIN_BATCH_SIZE=2, GRAD_ACC=1

######### Iter 1
DATA_DIR=iter1_max-car-8
# pi0-62pct_max-car-8
POLICY_MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# BoN_pi0_Q0-42pct-lr=5e-6_max-car-8
REWARD_MODEL="/share/portal/hw575/agent_prm/save/car_dealer/rm/250504_223511_iter1_max-car-8_pi0-62pct_max-car-8_peft=false/model/checkpoint-500"
MODEL_LOG_NAME="pi1_Q0-42pct-lr=5e-6_max-car-8"
# Manullay changing it
# 20000 / 3 / 2 / 1 = 3333
# 3333 / 5 = 555.5 --> 556
SAVE_FREQ=556
MAX_SEQ_LENGTH=3500
OUTPUT_LENGTH=400
GRAD_ACC=1
# TRAIN_BATCH_SIZE=2, GRAD_ACC=1


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
    --vllm_device cuda:3 \
    --vllm_gpu_memory_utilization 0.9 \
    --hf_metadata_dataset "" \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    --wandb_project_name "LLM_RM" \
    --with_tracking
