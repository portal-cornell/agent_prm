"""
Use case:

python scripts/eval/tools/plot_checkpoint_performance.py -n prm-pi1-q1-hd-with-original -b -bb
"""

import yaml
import json
import re
import os
import math
import pandas as pd
import numpy as np
from matplotlib import color_sequences
import matplotlib.pyplot as plt
import argparse

from agent_prm.utils.general_utils import load_json
from agent_prm.utils.cfg_utils import find_matching_iter

total_epochs = 1
baseline_to_name_in_csv = {
    "gpt-4o": "gpt4o",
    "base-3B": "3B",
    "pi0": "pi0-all-data-3epoches",
    "BoN_pi0_Q0-80pct-lr=5e-5": "BoN_pi0_Q0-80pct-lr=5e-5",
    "BoN_pi0_Q0-80pct-lr=5e-6": "BoN_pi0_Q0-80pct-lr=5e-6",
    "BoN_pi0_Q0*": "BoN_pi0_Q0-80pct-lr=5e-6",
    "pi1": "pi1-80pct_Q0-80pct-lr=5e-5",
    "BoN_pi1_Q1*": "BoN_pi1_Q1-60pct-lr=5e-6",
    "pi2": "pi2-60pct_Q1-60pct-lr=5e-6",
    "pi2-new-env": "pi2-60pct_Q1-60pct-lr=5e-6_new-env",
    "BoN_pi2_Q2*": "BoN_pi2_Q2-80pct-lr=5e-6",
    # Using hindsight to collect data for training PRMs
    "pi0-new-env": "pi0-new-env",
    "BoN_pi0_Q0*_pi0-new-env": "BoN_pi0_Q0-20pct-lr=5e-6_pi0-new-env",
    "BoN_pi0_Q0*_hindsight-redo": "BoN_pi0_Q0-20pct-lr=5e-6_hindsight-redo",
    "BoN_pi0_Q0*_hindsight-biased": "BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased",
    "pi1_hindsight": "pi1-77pct_Q0-80pct-lr=5e-6-hindsight",
    "BoN_pi1_Q1*_hindsight": "BoN_pi1_Q1-80pct-lr=5e-6_hindsight",
}  # in online_eval_table.csv

folder_to_baselines = {
    "prm-pi0-q0": ["gpt-4o", "base-3B", "pi0"],
    "prm-pi0-q0-best-rl": ["gpt-4o", "base-3B", "pi0"],
    "rl-pi1": ["gpt-4o", "pi0", "BoN_pi0_Q0*"],
    "prm-pi1-q1": ["gpt-4o", "pi0", "BoN_pi0_Q0*", "pi1"],
    "rl-pi2": ["gpt-4o", "pi0", "pi1", "BoN_pi1_Q1*"],
    "rl-pi2-with-4gpus": ["gpt-4o", "pi0", "pi1", "BoN_pi1_Q1*"],
    "prm-pi2-q2": ["gpt-4o", "pi0", "pi1", "pi2"],
    "rl-pi3": ["gpt-4o", "pi0", "pi1", "pi2", "BoN_pi2_Q2*"],
    ############################################################
    # Using hindsight to collect data for training PRMs
    "sft-pi0-new-env": ["gpt-4o", "base-3B", "pi0"],
    # Q0
    "prm-pi0-q0-hd": ["gpt-4o", "base-3B", "pi0"],
    "prm-pi0-q0-hd-with-original": ["gpt-4o", "base-3B", "pi0"],
    "prm-pi0-q0-hd-with-pi0-pi2-mix": ["gpt-4o", "base-3B", "pi0"],
    "prm-pi0-q0-hd-new-env": ["gpt-4o", "base-3B", "pi0-new-env", "pi2-new-env"],
    "prm-pi0-q0-hd-best": ["gpt-4o", "pi0-new-env", "pi2-new-env"],
    # pi1
    "rl-pi1-hd": ["gpt-4o", "pi0", "BoN_pi0_Q0*_hindsight"],
    "prm-pi1-q1-hd-with-original": ["gpt-4o", "pi0", "pi1_hindsight"],
    "rl-pi1-hd-new-env": ["gpt-4o", "pi0-new-env", "pi2-new-env", "BoN_pi0_Q0*_pi0-new-env", "BoN_pi0_Q0*_hindsight-redo", "BoN_pi0_Q0*_hindsight-biased"],
    "rl-pi1-hd-best": ["gpt-4o", "base-3B", "pi0-new-env", "pi2-new-env"],
    # pi2
    "rl-pi2-hd": ["gpt-4o", "pi0", "pi1", "BoN_pi1_Q1*_hindsight"],
    "rl-pi1-hd-with-pi0-pi2-mix": ["gpt-4o", "pi0"],
    "prm-pi2-q0-hd-with-pi0-pi2-mix": ["gpt-4o", "pi0", "pi1", "pi2"]
}

folder_to_regex = {
    "sft-pi0": [r'(pi0-(\d+)pct)-all-data-3epoches', r'pi0-all-data-3epoches'],
    # "prm-pi0-q0": [r'BoN-pi0\+(\d+)pct', r'BoN-(\d+)pct-balanced-Q', r'BoN-pi0\+Q0', r'BoN-balanced-Q'],
    "prm-pi0-q0": [r'BoN_pi0_(Q0-(\d+)pct)', r'BoN_pi0_Q0-lr'],
    "prm-pi0-q0-best": [r'BoN_pi0_(Q0-(\d+)pct)', r'BoN_pi0_Q0-lr'],
    "prm-pi0-q0-for-rl": [r'BoN_pi0_(Q0-(\d+)pct)', r'BoN_pi0_Q0-lr'],
    "prm-pi0-q0-best-rl": [r'BoN_pi0_(Q0-(\d+)pct)', r'BoN_pi0_Q0-lr'],
    "rl-pi1": [r'^(pi1-(\d+)pct)_Q0-80', r'^pi1_Q0-80'],
    "rl-pi1-with-best-pi0-bon": [r'^(pi1-(\d+)pct)_Q0-80', r'^pi1_Q0-80'],
    # 'rl-pi1-q0': [r'^pi1-(\d+)pct$', r'^pi1$', r'^BoN_pi1-(\d+)pct_Q0\*$', r'^BoN_pi1_Q0\*$']
    "prm-pi1-q1": [r'BoN_pi1_(Q1-(\d+)pct)', r'BoN_pi1_Q1-lr'],
    "rl-pi2": [r'^(pi2-(\d+)pct)_Q1-60', r'^pi2_Q1-60'],
    "rl-pi2-with-4gpus": [r'^(pi2-(\d+)pct)_Q1-60', r'^pi2_Q1-60'],
    "prm-pi2-q2": [r'BoN_pi2_(Q2-(\d+)pct)', r'BoN_pi2_Q2-lr'],
    "rl-pi3": [r'^(pi3-(\d+)pct)_Q2-80', r'^pi3_Q2-80'],
    ############################################################
    # Using hindsight to collect data for training PRMs
    "sft-pi0-new-env": [r'pi0-new-env'],
    # Q0
    "prm-pi0-q0-hd": 
        [r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight', r'BoN_pi0_Q0-lr=5e-6_hindsight'],
    "prm-pi0-q0-hd-with-original": 
        [r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6$', r'BoN_pi0_Q0-lr=5e-6$', r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight', r'BoN_pi0_Q0-lr=5e-6_hindsight'],
    "prm-pi0-q0-hd-with-pi0-pi2-mix": 
        [r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6$', r'BoN_pi0_Q0-lr=5e-6$', r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight', r'BoN_pi0_Q0-lr=5e-6_hindsight', r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_pi0-pi2-mix', r'BoN_pi0_Q0-lr=5e-6_pi0-pi2-mix'],
    "prm-pi0-q0-hd-new-env":
        [r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_pi0-new-env', r'BoN_pi0_Q0-lr=5e-6_pi0-new-env',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-redo', r'BoN_pi0_Q0-lr=5e-6_hindsight-redo',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-50-50', r'BoN_pi0_Q0-lr=5e-6_hindsight-50-50',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-30', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-30',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-50', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-50',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-70', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-70',
         ],
    "prm-pi0-q0-hd-best":
        [r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_pi0-new-env', r'BoN_pi0_Q0-lr=5e-6_pi0-new-env',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-redo', r'BoN_pi0_Q0-lr=5e-6_hindsight-redo',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-30', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-30',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-40', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-40',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-50', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-50',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-60', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-60',
         r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight-biased-on-70', r'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-70',
        ],
    # pi1
    "rl-pi1-hd":
        [r'^(pi1-(\d+)pct)_Q0-80pct-lr=5e-6-hindsight$', r'^pi1_Q0-80pct-lr=5e-6-hindsight$', r'^(pi1-(\d+)pct)_Q0-80pct-lr=5e-6$', r'^pi1_Q0-80pct-lr=5e-6$'],
    "rl-pi1-hd-with-pi0-pi2-mix":
        [
            r'^(pi1-(\d+)pct)_Q0-80pct-lr=5e-6$', r'^pi1_Q0-80pct-lr=5e-6$', 
            r'^(pi1-(\d+)pct)_Q0-80pct-lr=5e-6-hindsight$', r'^pi1_Q0-80pct-lr=5e-6-hindsight$', 
            r'^(pi1-(\d+)pct)_Q0-60pct-lr=5e-6-pi0-pi2-mix$', r'^pi1_Q0-60pct-lr=5e-6-pi0-pi2-mix$',
            r'^(pi2-(\d+)pct)_Q1-60', r'^pi2_Q1-60' # An upper bound on how much pi1-pi0-pi2-mix can improve
        ],
    "rl-pi1-hd-new-env":
        [
            r'^(pi1-(\d+)pct)_Q0-60pct-lr=5e-6_pi0-new-env$', r'^pi1_Q0-60pct-lr=5e-6_pi0-new-env$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_pi0-new-env$', r'^pi1_Q0-20pct-lr=5e-6_pi0-new-env$',
            r'^(pi1-(\d+)pct)_Q0-lr=5e-6_hindsight-redo$', r'^pi1_Q0-lr=5e-6_hindsight-redo$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_hindsight-redo$', r'^pi1_Q0-20pct-lr=5e-6_hindsight-redo$',
            r'^(pi1-(\d+)pct)_Q0-80pct-lr=5e-6_hindsight-50-50$', r'^pi1_Q0-80pct-lr=5e-6_hindsight-50-50$',
            r'^(pi1-(\d+)pct)_Q0-lr=5e-6_hindsight-biased$', r'^pi1_Q0-lr=5e-6_hindsight-biased$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_hindsight-biased$', r'^pi1_Q0-20pct-lr=5e-6_hindsight-biased$',
        ],
    "rl-pi1-hd-best":
        [
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_pi0-new-env$', r'^pi1_Q0-20pct-lr=5e-6_pi0-new-env$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_hindsight-redo$', r'^pi1_Q0-20pct-lr=5e-6_hindsight-redo$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_hindsight-biased$', r'^pi1_Q0-20pct-lr=5e-6_hindsight-biased$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_hindsight-biased-on-50$', r'^pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-50$',
            r'^(pi1-(\d+)pct)_Q0-20pct-lr=5e-6_hindsight-biased-on-60$', r'^pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-60$',
        ],
    # Q1
    "prm-pi1-q1-hd-with-original": 
        [r'BoN_pi1_(Q1-(\d+)pct)-lr=5e-6$', r'BoN_pi1_Q1-lr=5e-6$', r'BoN_pi1_(Q1-(\d+)pct)-lr=5e-6_hindsight', r'BoN_pi1_Q1-lr=5e-6_hindsight'],
    'rl-pi2-hd': 
        [r'^(pi2-(\d+)pct)_Q1-80pct-lr=5e-6-hindsight$', r'^pi2_Q1-80pct-lr=5e-6-hindsight$', r'^(pi2-(\d+)pct)_Q1-60pct-lr=5e-6$', r'^pi2_Q1-60pct-lr=5e-6$'],
    'prm-pi2-q0-hd-with-pi0-pi2-mix': 
        [r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_hindsight', r'BoN_pi0_Q0-lr=5e-6_hindsight', r'BoN_pi0_(Q0-(\d+)pct)-lr=5e-6_pi0-pi2-mix', r'BoN_pi0_Q0-lr=5e-6_pi0-pi2-mix', r'^BoN_pi2_(Q0-(\d+)pct)-lr=5e-6_hindsight-baseline$', r'^BoN_pi2_(Q0-(\d+)pct)-lr=5e-6_pi0-pi2-mix']
}

folder_to_plot_models = {
    "sft-pi0": ['gpt-4o', 'base-3B', 'pi0-lora', 'pi0-all-data-3epoches'],
    # "prm-pi0-q0": ['gpt-4o', 'pi0-lora', 'pi0', 'BoN-balanced-Q-3B-PSFT-all-data-3epoches', 'BoN-balanced-Q-lr=5e-6-3B-PSFT-all-data-3epoches', 'BoN-pi0+Q0', 'BoN-pi0+Q0-lr=5e-6'],
    "prm-pi0-q0": ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-5', 'BoN_pi0_Q0-lr=5e-6', 'BoN_pi0_Q0-lr=5e-7-no-reason', 'BoN_pi0_Q0-lr=5e-7'],
    "prm-pi0-q0-best": ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-5'],
    "prm-pi0-q0-for-rl": ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-5', 'BoN_pi0_Q0-lr=5e-6'],
    "prm-pi0-q0-best-rl": ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-6'],
    "rl-pi1": ['gpt-4o', 'pi0', 'pi0', 'pi1_Q0-80pct-lr=5e-5', 'pi1_Q0-80pct-lr=5e-6'],
    "rl-pi1-with-best-pi0-bon": ['gpt-4o', 'base-3B', 'pi0', 'pi1_Q0-80pct-lr=5e-5', 'pi1_Q0-80pct-lr=5e-6', 'BoN_pi0_Q0-80pct-lr=5e-5', 'BoN_pi0_Q0-80pct-lr=5e-6'],
    # "rl-pi1-q0": ['gpt-4o', 'pi0', 'pi0_Q0*', 'pi1', 'BoN_pi1_Q0*'],
    'prm-pi1-q1': ['gpt-4o', 'pi0', 'BoN_pi0_Q0*', 'pi1', 'BoN_pi1_Q1-lr=5e-6'],
    "rl-pi2": ['gpt-4o', 'pi0', 'pi1', 'BoN_pi1_Q1*', 'pi2_Q1-60pct-lr=5e-6'],
    "rl-pi2-with-4gpus": ['gpt-4o', 'pi0', 'pi1', 'BoN_pi1_Q1*', 'pi2_Q1-60pct-lr=5e-6', 'pi2_Q1-60pct-lr=5e-6_4gpus'],
    "prm-pi2-q2": ['gpt-4o', 'pi0', 'pi1', 'pi2', 'BoN_pi2_Q2-lr=5e-6'],
    "rl-pi3": ['gpt-4o', 'pi0', 'pi1', 'pi2', 'BoN_pi2_Q2*', 'pi3_Q2-80pct-lr=5e-6'],
    ############################################################
    # Using hindsight to collect data for training PRMs
    "sft-pi0-new-env": ['gpt-4o', 'base-3B', 'pi0', 'pi0-new-env'],
    # Q0
    "prm-pi0-q0-hd": 
        ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-6_hindsight-baseline', 'BoN_pi0_Q0-lr=5e-6_hindsight'],
    "prm-pi0-q0-hd-with-original": 
        ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-6', 'BoN_pi0_Q0-lr=5e-6_hindsight-baseline', 'BoN_pi0_Q0-lr=5e-6_hindsight'],
    "prm-pi0-q0-hd-with-pi0-pi2-mix": 
        ['gpt-4o', 'base-3B', 'pi0', 'BoN_pi0_Q0-lr=5e-6', 'BoN_pi0_Q0-lr=5e-6_hindsight-baseline', 'BoN_pi0_Q0-lr=5e-6_hindsight', 'BoN_pi0_Q0-lr=5e-6_pi0-pi2-mix'],
    "prm-pi0-q0-hd-new-env": 
        ['gpt-4o', 'base-3B', 'pi0-new-env', 'BoN_pi0_Q0-lr=5e-6_pi0-new-env', 'BoN_pi0_Q0-lr=5e-6_hindsight-redo', 
         'BoN_pi0_Q0-lr=5e-6_hindsight-50-50',
         'BoN_pi0_Q0-lr=5e-6_hindsight-biased',
         'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-30',
         'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-50'
        ],
    "prm-pi0-q0-hd-best":
        ['gpt-4o', 'pi0-new-env', 'pi2-new-env', 
         'BoN_pi0_Q0-lr=5e-6_pi0-new-env',
        #  'BoN_pi0_Q0-lr=5e-6_hindsight-redo', 
         'BoN_pi0_Q0-lr=5e-6_hindsight-biased',
        #  'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-30',
        #  'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-40',
         'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-50',
         'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-60',
        #  'BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-70'
        ],
    # pi1
    "rl-pi1-hd": 
        ['gpt-4o', 'pi0-new-env', 'BoN_pi0_Q0*_hindsight', 'pi1_Q0-80pct-lr=5e-6', 'pi1_Q0-80pct-lr=5e-6-hindsight'],
    "rl-pi1-hd-with-pi0-pi2-mix":
        ['gpt-4o', 'pi0-new-env', 'pi1_Q0-80pct-lr=5e-6', 'pi1_Q0-80pct-lr=5e-6-hindsight', 'pi1_Q0-60pct-lr=5e-6-pi0-pi2-mix', 'pi2_Q1-60pct-lr=5e-6'],
    "rl-pi1-hd-new-env":
        ['gpt-4o', 'pi0-new-env', 'pi2-new-env', 'BoN_pi0_Q0*_pi0-new-env', 'BoN_pi0_Q0*_hindsight-redo', 'BoN_pi0_Q0*_hindsight-biased', 'pi1_Q0-20pct-lr=5e-6_pi0-new-env', 'pi1_Q0-lr=5e-6_hindsight-redo', 'pi1_Q0-20pct-lr=5e-6_hindsight-redo', 'pi1_Q0-80pct-lr=5e-6_hindsight-50-50', 'pi1_Q0-lr=5e-6_hindsight-biased', 'pi1_Q0-20pct-lr=5e-6_hindsight-biased'],
    "rl-pi1-hd-best":
        ['gpt-4o', 'pi0-new-env', 'pi2-new-env', 'pi1_Q0-20pct-lr=5e-6_pi0-new-env',
        #  'pi1_Q0-20pct-lr=5e-6_hindsight-redo',
         'pi1_Q0-20pct-lr=5e-6_hindsight-biased',
         'pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-50',
         'pi1_Q0-20pct-lr=5e-6_hindsight-biased-on-60',
        ],
    # Q1
    "prm-pi1-q1-hd-with-original": 
        ['gpt-4o', 'pi0', 'pi1_hindsight', 'BoN_pi1_Q1-lr=5e-6', 'BoN_pi1_Q1-lr=5e-6_hindsight'],
    "rl-pi2-hd": 
        ['gpt-4o', 'pi0', 'pi1', 'BoN_pi1_Q1*_hindsight', 'pi2_Q1-60pct-lr=5e-6', 'pi2_Q1-80pct-lr=5e-6-hindsight'],
    'prm-pi2-q0-hd-with-pi0-pi2-mix': 
        ['gpt-4o', 'pi0', 'pi1', 'pi2', 'BoN_pi0_Q0-lr=5e-6_hindsight-baseline',  'BoN_pi0_Q0-lr=5e-6_pi0-pi2-mix', 'BoN_pi2_Q0-lr=5e-6_hindsight-baseline', 'BoN_pi2_Q0-lr=5e-6_pi0-pi2-mix']
}

EVAL_DIR = "data/twenty_questions/eval"
CSV_PATH = "data/twenty_questions/eval/online_eval_table.csv"
HINDSIHGT_CSV_PATH = "data/twenty_questions/eval/online_eval_table_hindsight.csv"
ROLLOUT_PER_TASK_DICT = {
    "train": 1,
    "val": 3,
    "test": 3
}

REWARD_MIN, REWARD_MAX = -18, -4
SUCCESS_RATE_MIN, SUCCESS_RATE_MAX = 0.4, 1.1

def is_valid_rollout(f: str, data_type: str) -> bool:
    """
    Check if the rollout is valid
    """
    is_a_rollout_file = f.endswith(".json") and not f.endswith("_summary_dict.json")
    to_include = False

    for i in range(ROLLOUT_PER_TASK_DICT[data_type]):
        if f"_{i}" in f:
            to_include = True
            break

    return is_a_rollout_file and to_include


def get_pct_of_training_progress(log_name: str, regex: str) -> int:
    """
    Parameters:
        log_name: str
            The log name of the model
                For example, if it's in progress: BoN-pi0+12pctQ0-lr=5e-6, 
                else if it's fully trained: BoN-pi0+Q0-lr=5e-6
    Returns:
        pct_of_training_progress: int
            The pct of training progress of the model
        class_name: str
            The class name of the model (for example: BoN-pi0+Q0-lr=5e-6)
    """
    if "pct" in regex:
        try:
            # 2 allows us to directly get the number
            pct = int(re.search(regex, log_name).group(2))

            part_with_pct = re.search(regex, log_name).group(1)
            part_with_pct_removed = part_with_pct.replace(f"-{pct}pct", "")

            class_name = log_name.replace(part_with_pct, part_with_pct_removed)

            return pct, class_name
        except Exception as e:
            print(f"Error:\n{e}")
            return 100, log_name
    else:
        # Assuming that it's fully trained
        return 100, log_name
    

def update_models_with_eval_results(models_to_plot: dict, baselines: list):
    for data_type in ["train", "val", "test"]:
        for class_name in models_to_plot:
            if class_name in baselines:
                continue

            for pct in models_to_plot[class_name]:
                agent_eval_dir = models_to_plot[class_name][pct]["agent_eval_dir"]

                if os.path.exists(os.path.join(agent_eval_dir, data_type)):
                    # Get all the rollouts that are used to consolidate the results
                    json_files = [f for f in os.listdir(os.path.join(agent_eval_dir, data_type)) if is_valid_rollout(f, data_type)]

                    if not any([f.endswith(f"{ROLLOUT_PER_TASK_DICT[data_type]-1}.json") for f in json_files]):
                        print(f"WARNING: {agent_name} does not have all the rollouts for {data_type}, which needs {ROLLOUT_PER_TASK_DICT[data_type]} rollouts per task")
                        # input("Press Enter to continue...")

                    if len(json_files) > 0:
                        # Compute rewards efficiently
                        all_rewards = [sum(t["reward"] for t in load_json(os.path.join(agent_eval_dir, data_type, f))) for f in json_files]
                        mean_reward = np.mean(all_rewards)
                        se_reward = np.std(all_rewards)/math.sqrt(len(all_rewards))

                        # Compute success rate
                        all_success_rates = [load_json(os.path.join(agent_eval_dir, data_type, f))[-1]["reward"] == 0 for f in json_files]
                        mean_success_rate = np.mean(all_success_rates)
                        se_success_rate = np.std(all_success_rates)/math.sqrt(len(all_success_rates))

                        models_to_plot[class_name][pct][data_type] = {
                            "mean_reward": mean_reward,
                            "se_reward": se_reward,
                            "mean_success_rate": mean_success_rate,
                            "se_success_rate": se_success_rate
                        }
                    else:
                        models_to_plot[class_name][pct][data_type] = {
                            "mean_reward": REWARD_MIN,
                            "se_reward": 0,
                            "mean_success_rate": SUCCESS_RATE_MIN,
                            "se_success_rate": 0
                        }
                else:
                    print(f"WARNING: {agent_name} does not have any rollouts for {data_type}")
                    models_to_plot[class_name][pct][data_type] = {
                        "mean_reward": None,
                        "se_reward": None,
                        "mean_success_rate": None,
                        "se_success_rate": None
                    }


def plot_models(models_to_plot: dict, plot_path: str, models_to_plot_names: list, baselines: list, error_bar: bool = False, error_bar_baseline: bool = False):
    """
    Plot the models and save the models at 
    """
    data_types = ['train', 'val', 'test']
    colors = color_sequences['Dark2'] + color_sequences['Set2']
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 8))

    linewidth = 3
    markersize = 10
    fontsize = 16

    # Plot reward metrics
    for i, data_type in enumerate(data_types):
        ax = axes[0, i]
        for j, model in enumerate(models_to_plot_names):
            if model in baselines:
                if models_to_plot[model][data_type]["mean_reward"] is not None:
                    # Draw a horizontal line with standard error
                    ax.axhline(y=models_to_plot[model][data_type]["mean_reward"], color=colors[j], linestyle='--', label=model, linewidth=linewidth)

                    # Draw standard error
                    if error_bar_baseline:
                        ax.fill_between(
                            [0, total_epochs],
                            models_to_plot[model][data_type]["mean_reward"] - models_to_plot[model][data_type]["se_reward"],
                            models_to_plot[model][data_type]["mean_reward"] + models_to_plot[model][data_type]["se_reward"],
                            color=colors[j], alpha=0.2)
            else:
                epochs = np.array([int(pct)/100.0*total_epochs for pct in models_to_plot[model].keys()])
                rewards = np.array([models_to_plot[model][pct][data_type]["mean_reward"] for pct in models_to_plot[model].keys()])
                se_rewards = np.array([models_to_plot[model][pct][data_type]["se_reward"] for pct in models_to_plot[model].keys()])

                ax.plot(epochs, rewards, label=model, color=colors[j], marker='o', linewidth=linewidth, markersize=markersize)
                if error_bar:
                    ax.fill_between(epochs, rewards - se_rewards, 
                                    rewards + se_rewards, color=colors[j], alpha=0.2)
            
        # Set the y axis range to be -20 and 0
        ax.set_ylim(REWARD_MIN, REWARD_MAX)
        ax.set_xticks(np.arange(0, total_epochs+total_epochs/10, total_epochs/10))
        ax.set_title(f'Reward - {data_type}', fontsize=fontsize)
        ax.set_xlabel('Epochs', fontsize=fontsize)
        ax.set_ylabel('Reward', fontsize=fontsize)

    # Plot success metrics
    for i, data_type in enumerate(data_types):
        ax = axes[1, i]
        for j, model in enumerate(models_to_plot_names):
            if model in baselines:
                if models_to_plot[model][data_type]["mean_success_rate"] is not None:
                    # Draw a horizontal line with standard error
                    ax.axhline(y=models_to_plot[model][data_type]["mean_success_rate"], color=colors[j], linestyle='--', label=model, linewidth=linewidth)

                    # Draw standard error
                    if error_bar_baseline:
                        ax.fill_between(
                            [0, total_epochs],
                            models_to_plot[model][data_type]["mean_success_rate"] - models_to_plot[model][data_type]["se_success_rate"],
                            models_to_plot[model][data_type]["mean_success_rate"] + models_to_plot[model][data_type]["se_success_rate"],
                            color=colors[j], alpha=0.2)
            else:
                # Calculate the epoch (x-axis)
                epochs = np.array([int(pct)/100.0*total_epochs for pct in models_to_plot[model].keys()])

                # Success rate
                success_rates = np.array([models_to_plot[model][pct][data_type]["mean_success_rate"] for pct in models_to_plot[model].keys()])
                se_success_rates = np.array([models_to_plot[model][pct][data_type]["se_success_rate"] for pct in models_to_plot[model].keys()])
                
                ax.plot(epochs, success_rates, label=model, color=colors[j], marker='o', linewidth=linewidth, markersize=markersize)
                if error_bar:
                    ax.fill_between(epochs, success_rates - se_success_rates, 
                                    success_rates + se_success_rates, color=colors[j], alpha=0.2)
        
        # Set the y axis range to be 0 and 1
        ax.set_ylim(SUCCESS_RATE_MIN, SUCCESS_RATE_MAX)
        ax.set_xticks(np.arange(0, total_epochs+total_epochs/10, total_epochs/10))
        ax.set_title(f'Success - {data_type}', fontsize=fontsize)
        ax.set_xlabel('Epochs', fontsize=fontsize)
        ax.set_ylabel('Success Rate', fontsize=fontsize)


    # Add one legend for all plots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fancybox=True, shadow=False, framealpha=1.0, fontsize=fontsize)

    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plot_path)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True, help="The name of folder to save the plot")
    parser.add_argument("-e", "--use_existing_table", action="store_true")
    parser.add_argument("-b", "--error_bar", action="store_true")
    parser.add_argument("-bb", "--error_bar_baseline", action="store_true")
    args = parser.parse_args()

    save_path = os.path.join("playground/eval", args.name)
    os.makedirs(save_path, exist_ok=True)

    try:
        regex_to_use = folder_to_regex[args.name]
        models_to_plot_names = folder_to_plot_models[args.name]
        baselines = folder_to_baselines[args.name]
    except KeyError:
        raise ValueError(f"Invalid name: {args.name}, available names are: {folder_to_regex.keys()}")

    # Load the yaml file
    with open("configs/eval_config/twenty_questions.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not args.use_existing_table:
        # Get the models to plot
        """
        Format:
        [
            "name_of_this_class_of_models": {
                "pct_of_training_progress": "path_to_eval_dir",
                ...
            }
        ]
        """

        # Get the models to plot
        models_to_plot = {}

        # Load the current CSV for baselines
        if "hd" in args.name:
            df = pd.read_csv(HINDSIHGT_CSV_PATH)
        else:
            df = pd.read_csv(CSV_PATH)

        for baseline_plot_name in baselines:
            models_to_plot[baseline_plot_name] = {}
            # Get it from the CSV
            for data_type in ["train", "val", "test"]:
                baseline_csv_name = baseline_to_name_in_csv[baseline_plot_name]
                models_to_plot[baseline_plot_name][data_type] = {
                    "mean_reward": df[df["model"] == baseline_csv_name][f"{data_type} (avg reward)"].values[0],
                    "se_reward": df[df["model"] == baseline_csv_name][f"{data_type} (se reward)"].values[0],
                    "mean_success_rate": df[df["model"] == baseline_csv_name][f"{data_type} (avg success rate)"].values[0],
                    "se_success_rate": df[df["model"] == baseline_csv_name][f"{data_type} (se success rate)"].values[0]
                }

        for agent_config in cfg["agents"]:
            for regex in regex_to_use:
                if re.search(regex, agent_config["log_name"]):
                    pct, class_name = get_pct_of_training_progress(agent_config["log_name"], regex)

                    agent_name = agent_config["model_id"] if agent_config["type"] != "best_of_n" else agent_config["generator"]["model_id"]
                        
                    if "checkpoint" in agent_name:
                        agent_name = os.path.basename(agent_name.split("/")[-2])
                    else:
                        agent_name = os.path.basename(agent_name)

                    agent_name = f"{agent_config['log_name']}_{agent_name}"

                    agent_eval_dir = os.path.join(EVAL_DIR, find_matching_iter(agent_config['log_name']), agent_name)

                    if class_name not in models_to_plot:
                        models_to_plot[class_name] = {}

                    models_to_plot[class_name][pct] = {
                        "agent_eval_dir": agent_eval_dir
                    }

        with open(os.path.join(save_path, "models_to_plot.json"), "w") as f:
            json.dump(models_to_plot, f, indent=4)

        print(json.dumps(models_to_plot, indent=4))
        input("Press Enter to continue...")

        print("Updating models with eval results")
        update_models_with_eval_results(models_to_plot, baselines)

        with open(os.path.join(save_path, "models_to_plot_with_eval_results.json"), "w") as f:
            json.dump(models_to_plot, f, indent=4)
    else:
        with open(os.path.join(save_path, "models_to_plot_with_eval_results.json"), "r") as f:
            models_to_plot = json.load(f)

    print("Plotting models")
    plot_models(models_to_plot, os.path.join(save_path, f"{args.name}_plot.png"), models_to_plot_names, baselines, args.error_bar, args.error_bar_baseline)