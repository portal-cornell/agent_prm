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

DOMAIN=twenty_questions # alfworld, twenty_questions
SAVE_FREQ=250
MAX_SEQ_LEN=2048

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


# #### with hindsight (biased filtering - 20-80 onpolicy/offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-20
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-20"

# #### with hindsight (biased filtering - 80-20 onpolicy/offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-80
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-80"

# #### with hindsight (biased filtering - 30-70 onpolicy/offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-30
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-30"

#### with hindsight (biased filtering - 70-30 onpolicy/offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-70
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-70"


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


# # #### with hindsight (biased filtering - 0-100 onpolicy/offpolicy)
# DATA_DIR=iter2_hindsight-biased
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_iter2_hindsight-biased"

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

# # #### with hindsight (biased filtering - 0-100 onpolicy/offpolicy)
# DATA_DIR=iter3_hindsight-biased
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_iter3_hindsight-biased"

################ iter1 other exploration baselines
# ######### with pi*
# DATA_DIR=iter1_best-pi-on-60
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_best-pi-on-60"
######### with explorative pi0
# DATA_DIR=iter1_explorative-pi-on-60
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_explorative-pi-on-60"
######## with high-temp pi0
# DATA_DIR=iter1_high-temp-on-60
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_high-temp-pi-on-60"

######## with high-temp pi0 (not follow our data collection)
DATA_DIR=iter1_pi0-new-env_high-temp
MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
MODEL_LOG_NAME="pi0_high-temp-pi-on-60"
# 10000 / 4 /  4 = 625
# 625 / 5 = 125
SAVE_FREQ=125

############################################################## exploration during RL
# Because now we have to track the summary, we need to train a slightly different q0
# DATA_DIR=iter1_hindsight-biased-on-60_with-summary
# MODEL="/share/portal/hw575/agent_prm/save/twenty_questions/sft/250307_212417_iter0-all_meta-llama-Llama-3.2-3B-Instruct_peft=false_epoch3+all/checkpoint-120"
# MODEL_LOG_NAME="pi0_hindsight-biased-on-60_with-summary"

#=====================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ CAR DEALER
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#=====================================================================================================
# DOMAIN=car_dealer
# SAVE_FREQ=500
# MAX_SEQ_LEN=3500

# #################################### PRM + RL
# ########### iter1
# DATA_DIR=iter1
# # pi0-83pct
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250430_180817_iter0_pi0_vanilla_epochs=3/checkpoint-124"
# MODEL_LOG_NAME="pi0-83pct"

# # iter1 max-car-8 (to reduce max seq len)
# DATA_DIR=iter1_max-car-8
# # pi0-62pct_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# MODEL_LOG_NAME="pi0-62pct_max-car-8"

# ########### iter2
# DATA_DIR=iter2_max-car-8
# # pi1_Q0-42pct-lr=5e-6_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250505_222544_iter1_max-car-8_pi1_Q0-42pct-lr=5e-6_max-car-8"
# MODEL_LOG_NAME="pi1_Q0-42pct-lr=5e-6_max-car-8"
# NUM_PROCESSES=4
# SAVE_FREQ=236

########### iter3
# DATA_DIR=iter3_max-car-8
# # pi2_Q1-60pct-lr=5e-6_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250508_010928_iter2_max-car-8_pi2_Q1-60pct-lr=5e-6_max-car-8"
# MODEL_LOG_NAME="pi2_Q1-60pct-lr=5e-6_max-car-8"
# NUM_PROCESSES=4
# SAVE_FREQ=236

# #################################### Hindsight
###### iter 1 - 40 pct onpolicy, 60 pct offpolicy
# DATA_DIR=iter1_hindsight-biased-on-40
# # pi0-62pct_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# MODEL_LOG_NAME="pi0-62pct_max-car-8_hindsight-biased-on-40"

# ###### iter 1 - 50 pct onpolicy, 50 pct offpolicy
# DATA_DIR=iter1_hindsight-biased-on-50
# # pi0-62pct_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# MODEL_LOG_NAME="pi0-62pct_max-car-8_hindsight-biased-on-50"

# ###### iter 1 - 60 pct onpolicy, 40 pct offpolicy
# DATA_DIR=iter1_hindsight-biased-on-60
# # pi0-62pct_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# MODEL_LOG_NAME="pi0-62pct_max-car-8_hindsight-biased-on-60"

# # ###### iter 1 - 0 pct onpolicy, 100 pct offpolicy
# DATA_DIR=iter1_hindsight-biased
# # pi0-62pct_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# MODEL_LOG_NAME="pi0-62pct_max-car-8_hindsight-biased"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

# # ###### iter 2 - 50 pct onpolicy, 50 pct offpolicy (Made a mistake to incorporate past data)
# DATA_DIR=iter2_hindsight-biased-on-50
# # pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250506_185257_iter1_hindsight-biased-on-50_pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50"
# MODEL_LOG_NAME="pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

# ###### iter 2 - 60 pct onpolicy, 40 pct offpolicy (Made a mistake to incorporate past data)
# DATA_DIR=iter2_hindsight-biased-on-60
# # pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250506_185257_iter1_hindsight-biased-on-50_pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50"
# MODEL_LOG_NAME="pi1-hd-50_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-60"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

# # ###### iter 2 - 50 pct onpolicy, 50 pct offpolicy (Made a mistake to incorporate past data)
# DATA_DIR=iter2_hindsight-biased-on-50_no-past-rollout
# # pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250506_185257_iter1_hindsight-biased-on-50_pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50"
# MODEL_LOG_NAME="pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50_no-past-rollout"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

# # ###### iter 2 - 60 pct onpolicy, 40 pct offpolicy (Made a mistake to incorporate past data)
# DATA_DIR=iter2_hindsight-biased-on-60_no-past-rollout
# # pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250506_185257_iter1_hindsight-biased-on-50_pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-50"
# MODEL_LOG_NAME="pi1_Q0-85pct-lr=5e-6_max-car-8_hindsight-biased-on-60_no-past-rollout"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

# ###### iter 2 - 0 pct onpolicy, 100 pct offpolicy
# DATA_DIR=iter2_hindsight-biased_past-scale
# # pi0-62pct_max-car-8
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/sft/250504_061410_iter0-max-car-8_pi0_vanilla_max-car-8_epochs=3/checkpoint-93"
# MODEL_LOG_NAME="pi0-62pct_max-car-8_hindsight-biased_past-scale"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

# ###### iter 3 - 50 pct onpolicy, 50 pct offpolicy
# DATA_DIR=iter3_hindsight-biased-on-50_past-scale
# # pi2-80pct_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50
# MODEL="/share/portal/hw575/agent_prm/save/car_dealer/online_dpo/250510_115714_iter2_hindsight-biased-on-50_no-past-rollout_pi2_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50/checkpoint-800"
# MODEL_LOG_NAME="pi2-80pct_Q1-85pct-lr=5e-6_no-past_hindsight-biased-on-50_past-scale"
# NUM_PROCESSES=4
# NUM_EVALS=10
# # 20000 / 4 / 4 = 1250
# # 1250 / 5 = 250
# SAVE_FREQ=250

#=====================================================================================================
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ GUESS MY CITY
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#=====================================================================================================
# DOMAIN=guess_my_city
# SAVE_FREQ=250
# MAX_SEQ_LEN=3000

# # #################################### PRM + RL
# ########### iter1
# DATA_DIR=iter1
# # pi0-82pct
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/sft/250506_225348_iter0_pi0_vanilla_epochs=3/checkpoint-56"
# MODEL_LOG_NAME="pi0-82pct"

# # ########### iter2
# DATA_DIR=iter2
# # pi1-41pct_Q0-60pct-lr=5e-6
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/online_dpo/250509_150054_iter1_pi1_Q0-60pct-lr=5e-6/checkpoint-332"
# MODEL_LOG_NAME="pi1-41pct_Q0-60pct-lr=5e-6"

# ########### iter3
# DATA_DIR=iter3
# # pi2_Q0-60pct-lr=5e-6
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/online_dpo/250512_195327_iter2_pi1_Q1-lr=5e-6"
# MODEL_LOG_NAME="pi2_Q0-60pct-lr=5e-6"

# # #################################### PRM + RL
# ########### iter1 (50% onpolicy, 50% offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-50
# # pi0-82pct
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/sft/250506_225348_iter0_pi0_vanilla_epochs=3/checkpoint-56"
# MODEL_LOG_NAME="pi0-82pct_hindsight-biased-on-50"

# ########### iter1 (60% onpolicy, 40% offpolicy)
# DATA_DIR=iter1_hindsight-biased-on-60
# # pi0-82pct
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/sft/250506_225348_iter0_pi0_vanilla_epochs=3/checkpoint-56"
# MODEL_LOG_NAME="pi0-82pct_hindsight-biased-on-60"

# # ########### iter1 (50% onpolicy, 50% offpolicy) - A simpler way to estimate reward
# DATA_DIR=iter1_hindsight-biased-on-50_from-sparse-r
# # pi0-82pct
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/sft/250506_225348_iter0_pi0_vanilla_epochs=3/checkpoint-56"
# MODEL_LOG_NAME="pi0-82pct_hindsight-biased-on-50-from-sparse-r"

# # ########### iter1 (60% onpolicy, 40% offpolicy) - A simpler way to estimate reward
# DATA_DIR=iter1_hindsight-biased-on-60_from-sparse-r
# # pi0-82pct
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/sft/250506_225348_iter0_pi0_vanilla_epochs=3/checkpoint-56"
# MODEL_LOG_NAME="pi0-82pct_hindsight-biased-on-60-from-sparse-r"

# # ########### iter1 (0% onpolicy, 100% offpolicy) - A simpler way to estimate reward
# DATA_DIR=iter1_hindsight-biased_from-sparse-r
# # pi0-82pct
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/sft/250506_225348_iter0_pi0_vanilla_epochs=3/checkpoint-56"
# MODEL_LOG_NAME="pi0-82pct_hindsight-biased_from-sparse-r"

# # ########### iter2 (50% onpolicy, 50% offpolicy) - A simpler way to estimate reward
# DATA_DIR=iter2_hindsight-biased-on-50_from-sparse-r
# # pi1-60pct_Q0-60pct-lr=5e-6_highsight-biased-on-50-from-sparse-r_retry2
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/online_dpo/250513_124139_iter1_hindsight-biased-on-50_from-sparse-r_pi1_Q0-60pct-lr=5e-6_hindsight-biased-on-50-from-sparse-r/checkpoint-996"
# MODEL_LOG_NAME="pi1-41pct_Q0-60pct-lr=5e-6_hindsight-biased-on-50-from-sparse-r"

# ########### iter2 (60% onpolicy, 40% offpolicy) - A simpler way to estimate reward
# DATA_DIR=iter2_hindsight-biased-on-60_from-sparse-r
# # pi1-60pct_Q0-60pct-lr=5e-6_highsight-biased-on-60-from-sparse-r_retry2
# MODEL="/share/portal/hw575/agent_prm/save/guess_my_city/online_dpo/250513_124139_iter1_hindsight-biased-on-50_from-sparse-r_pi1_Q0-60pct-lr=5e-6_hindsight-biased-on-50-from-sparse-r/checkpoint-996"
# MODEL_LOG_NAME="pi1-41pct_Q0-60pct-lr=5e-6_hindsight-biased-on-60-from-sparse-r"


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

export TRITON_CACHE_DIR=/share/portal/hw575
TRITON_CACHE_DIR=/share/portal/hw575/.triton

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
