"""
python scripts/eval/tools/plot_overall_barplot.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import color_sequences
from datetime import datetime
import os

COLORS = [] # Default values
##################################### 20 questions #####################################
# ##### Original
# REWARD_MIN = 0
# REWARD_MAX = 10
# SUCCESS_RATE_MIN = 0
# SUCCESS_RATE_MAX = 1

# DOMAIN = "twenty_questions"
# TABLE_SUFFIX = ""
# EVAL_CSV_PATH = f"data/{DOMAIN}/eval/online_eval_table{TABLE_SUFFIX}.csv"

# model_to_name_in_csv = {
#     "gpt4o": "gpt4o",
#     "Llama3.2-3B": "3B",
#     "Llama3.2-3B π0": "pi0-all-data-3epoches",
#     "Llama3.2-3B BoN(π0, Q0)": "BoN_pi0_Q0-80pct-lr=5e-6",
#     "Llama3.2-3B π1": "pi1-80pct_Q0-80pct-lr=5e-5",
#     "Llama3.2-3B BoN(π1, Q1)": "BoN_pi1_Q1-60pct-lr=5e-6",
#     "Llama3.2-3B π2": "pi2-60pct_Q1-60pct-lr=5e-6",
#     "Llama3.2-3B BoN(π2, Q2)": "BoN_pi2_Q2-80pct-lr=5e-6",
#     "Llama3.2-3B π3": "pi3-38pct_Q2-80pct-lr=5e-6"
# }

# ###### Hindisght PRM
REWARD_MIN = 0
REWARD_MAX = 15
SUCCESS_RATE_MIN = 0
SUCCESS_RATE_MAX = 1

DOMAIN = "twenty_questions"
TABLE_SUFFIX = "_hindsight"
EVAL_CSV_PATH = f"data/{DOMAIN}/eval/online_eval_table{TABLE_SUFFIX}.csv"

# q0
# model_to_name_in_csv = {
#     "gpt4o": "gpt4o",
#     # "3B": "3B",
#     "3B π0": "pi0-new-env",
#     "3B π1": "pi1_Q0-60pct-lr=5e-6_pi0-new-env",
#     "3B π2": "pi2_Q1-40pct-lr=5e-6_new-env_from-pi0",
#     "3B BoN(π0, Q0) 100-off-0-on": "BoN_pi0_Q0-20pct-lr=5e-6_hindsight-biased",
#     # "3B BoN(π0, Q0) 70-off-30-on": "BoN_pi0_Q0-40pct-lr=5e-6_hindsight-biased-on-30",
#     "3B BoN(π0, Q0) 60-off-40-on": "BoN_pi0_Q0-80pct-lr=5e-6_hindsight-biased-on-40",
#     "3B BoN(π0, Q0) 50-off-50-on": "BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-50",
#     "3B BoN(π0, Q0) 40-off-60-on": "BoN_pi0_Q0-lr=5e-6_hindsight-biased-on-60",
#     # "3B BoN(π0, Q0) 30-off-70-on": "BoN_pi0_Q0-40pct-lr=5e-6_hindsight-biased-on-70",
#     "3B BoN(π0, Q0) 0-off-100-on": "BoN_pi0_Q0-60pct-lr=5e-6_pi0-new-env",
# }

# pi1
# model_to_name_in_csv = {
#     "gpt4o": "gpt4o",
#     # "3B": "3B",
#     "3B π0": "pi0-new-env",
#     "3B π1": "pi1_Q0-60pct-lr=5e-6_pi0-new-env",
#     "3B π2": "pi2_Q1-40pct-lr=5e-6_new-env_from-pi0",
#     "3B π1 60-off-40-on": "pi1-20pct_Q0-80pct-lr=5e-6_hindsight-biased-on-40",
#     "3B π1 40-off-60-on": "pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60",
# }

# q1
# model_to_name_in_csv = {
#     "gpt4o": "gpt4o",
#     # "3B": "3B",
#     "3B π0": "pi0-new-env",
#     "3B π1": "pi1_Q0-60pct-lr=5e-6_pi0-new-env",
#     "3B π2": "pi2_Q1-40pct-lr=5e-6_new-env_from-pi0",
#     "3B BoN(π1, Q1) 0-off-100-on": "BoN_pi1_Q1-40pct-lr=5e-6_new-env",
#     "3B π1 40-off-60-on": "pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60",
#     "3B BoN(π1, Q1) 40-off-60-on": "BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60",
# }

# q1 (the not so important need to train from pi0)
# model_to_name_in_csv = {
#     "3B π0": "pi0-new-env",
#     "3B BoN(π1, Q1 from π0) 40-off-60-on": "BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60_from-pi0",
#     "3B BoN(π1, Q1 from π1) 40-off-60-on": "BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60",
#     "3B π1 40-off-60-on": "pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60",
# }

# pi2
# model_to_name_in_csv = {
#     "gpt4o": "gpt4o",
#     # "3B": "3B",
#     "3B π0": "pi0-new-env",
#     "3B π1": "pi1_Q0-60pct-lr=5e-6_pi0-new-env",
#     "3B π2": "pi2_Q1-40pct-lr=5e-6_new-env_from-pi0",
#     "3B π1 40-off-60-on": "pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60",
#     "3B BoN(π1, Q1) 40-off-60-on": "BoN_pi1_Q1-lr=5e-6_hindsight-biased-on-60",
#     "3B π2 40-off-60-on": "pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0",
# }

# # pi2 (the importance to train from pi0)
# model_to_name_in_csv = {
#     # "gpt4o": "gpt4o",
#     # "3B": "3B",
#     "3B π0": "pi0-new-env",
#     "3B π1": "pi1_Q0-60pct-lr=5e-6_pi0-new-env",
#     "3B π2 (from π1)": "pi2-40pct_Q1-40pct-lr=5e-6_new-env",
#     "3B π2 (from π0)": "pi2_Q1-40pct-lr=5e-6_new-env_from-pi0",
# }


# Baseline plot
# COLORS = ['#EF772B', '#5B5B5B', '#a6a6a6']
# model_to_name_in_csv = {
#     "gpt4o": "gpt4o",
#     "3B": "3B",
#     "3B π0": "pi0-new-env"
# }


# # Vanilla RL plot
# COLORS = ['#a6a6a6', '#ea61a3', '#e3247f', '#b51763']
# model_to_name_in_csv = {
#     "3B π0": "pi0-new-env",
#     "3B π1": "pi1_Q0-60pct-lr=5e-6_pi0-new-env",
#     "3B π2": "pi2_Q1-40pct-lr=5e-6_new-env_from-pi0",
#     "3B π3": "pi3-80pct_Q2-lr=5e-6_new-env-pi2-from-pi0_from-pi0",
# }

# Hindsight PRM plot
# COLORS = ['#a6a6a6', '#abdea1', '#83CE74', '#abdea1']
# model_to_name_in_csv = {
#     "3B π0": "pi0-new-env",
#     "3B π1 (Hindsight)": "pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60",
#     "3B π2 (Hindsight)": "pi2_Q1-60pct-lr=5e-6_hindsight-biased-on-60_from-pi0",
#     "3B π3 (Hindsight)": "pi3-60pct_Q2-lr=5e-6_hindsight-biased-on-60",
# }

# LEAP plot
# COLORS = ['#a6a6a6', '#a892d3', '#7D5CBD', '#b6a4da']
# model_to_name_in_csv = {
#     "3B π0": "pi0-new-env",
#     "3B π1 (LEAP)": "pi1-44pct_leap_from-base-3B_1epoch",
#     "3B π2 (LEAP)": "pi2-67pct_leap_from-base-3B_1epoch",
#     "3B π3 (LEAP)": "pi3-67pct_leap_from-base-3B_1epoch",
# }

# # Multi-STaR plot
# COLORS = ['#a6a6a6', '#b3d9ff', '#69B3FF', '#99ccff']
# model_to_name_in_csv = {
#     "3B π0": "pi0-new-env",
#     "3B π1 (Multi-STaR)": "pi1-41pct_multi-star_from-base-3B_1epoch_10k-data_lr=3e-6",
#     "3B π2 (Multi-STaR)": "pi2-62pct_multi-star_from-base-3B_1epoch_10k-data-mix-50pct-past_lr=3e-6",
#     "3B π3 (Multi-STaR)": "pi3-82pct_multi-star_from-base-3B_1epoch_10k-data-mix-50pct-past_lr=3e-6",
# }

# Exploration Strategy Plot
COLORS = ['#a6a6a6', '#ffe43d', '#e6c700', '#abdea1']
model_to_name_in_csv = {
    "3B π0": "pi0-new-env",
    "3B π1 (π0 to explore)": "pi1-40pct_Q0-80pct-lr=5e-6_explorative-pi-on-60",
    "3B π1 (π* to explore)": "pi1-25pct_Q0-lr=5e-6_best-pi-on-60",
    "3B π1 (Hindsight)": "pi1-60pct_Q0-lr=5e-6_hindsight-biased-on-60",
}


##################################### Car Dealer #####################################
# REWARD_MIN = 0
# REWARD_MAX = 1.5
# SUCCESS_RATE_MIN = 0
# SUCCESS_RATE_MAX = 1

# DOMAIN = "car_dealer"
# TABLE_SUFFIX = ""
# EVAL_CSV_PATH = f"data/{DOMAIN}/eval/online_eval_table{TABLE_SUFFIX}.csv"

# model_to_name_in_csv = {
#     "gpt4o": "gpt4o"
# }

# Create the inverse mapping from model_to_name_in_csv
name_to_model_in_csv = {v: k for k, v in model_to_name_in_csv.items()}

# Load the overall eval CSV
df = pd.read_csv(EVAL_CSV_PATH)

# Create the sub-df for the models to plot (Warning, the order of the model is based on the order of the model in the csv file)
df_to_plot = df[df['model'].isin(model_to_name_in_csv.values())]

# Create the mapping first
df_to_plot['model'] = df_to_plot['model'].map(name_to_model_in_csv)

# Then create categorical with the correct order
ordered_models = list(model_to_name_in_csv.keys())  # Gets the keys in the original order
df_to_plot['model'] = pd.Categorical(df_to_plot['model'], categories=ordered_models, ordered=True)

# Sort by the categorical column
df_to_plot = df_to_plot.sort_values('model')

# Define columns for plotting
categories = ['train', 'val', 'test', 'total']
metrics = ['avg reward', 'avg success rate']
errors = ['se reward', 'se success rate']

# Number of models
models = df_to_plot['model']
x = np.arange(len(models))  # The label locations

# Create the figure and axes
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)

# Define bar width
bar_width = 0.6

# Define color sequence
if COLORS == []:
    colors = color_sequences['Dark2'] + color_sequences['Accent']
else:
    colors = COLORS

# Loop through metrics and categories
for row_idx, metric in enumerate(metrics):
    for col_idx, category in enumerate(categories):
        ax = axes[row_idx, col_idx]
        
        # Extract data
        avg_values = df_to_plot[f'{category} ({metric})']
        if metric == 'avg reward' and DOMAIN == "twenty_questions":
            avg_values = 20 + avg_values # Shift the reward to positive scale
        se_values = df_to_plot[f'{category} ({errors[row_idx]})']
        
        # Plot bar chart with error bars and colors for each model
        for i, (value, error) in enumerate(zip(avg_values, se_values)):
            bar = ax.bar(x[i], value, yerr=error, capsize=5, alpha=0.7, width=bar_width, color=colors[i])
            # Add text label above each bar
            ax.text(x[i], value + error, f'{value:.2f}', 
                   ha='center', va='bottom')
        
        # Set labels and title
        ax.set_title(f'{category.capitalize()} - {metric.capitalize()}')
        if col_idx == 0:
            ax.set_ylabel(metric.capitalize())
        
        # Set x-ticks only for the last row
        if row_idx == 1:
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])

        # Set the range for y-axis
        if metric == 'avg reward':
            ax.set_ylim(REWARD_MIN, REWARD_MAX)
        elif metric == 'avg success rate':
            ax.set_ylim(SUCCESS_RATE_MIN, SUCCESS_RATE_MAX)


# Adjust layout and show plot
plt.tight_layout()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the plot to a file
os.makedirs(f'playground/{DOMAIN}/barplot', exist_ok=True)
plt.savefig(f'playground/{DOMAIN}/barplot/barplot_{current_time}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'playground/{DOMAIN}/barplot/_barplot.png', dpi=300, bbox_inches='tight') # Have an accessible copy that stays at the top