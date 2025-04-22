import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import color_sequences
from datetime import datetime
REWARD_MIN = 0
REWARD_MAX = 10
SUCCESS_RATE_MIN = 0
SUCCESS_RATE_MAX = 1

EVAL_CSV_PATH = "data/twenty_questions/eval/online_eval_table.csv"

model_to_name_in_csv = {
    "gpt4o": "gpt4o",
    "Llama3.2-3B": "3B",
    "Llama3.2-3B π0": "pi0-all-data-3epoches",
    "Llama3.2-3B BoN(π0, Q0)": "BoN_pi0_Q0-80pct-lr=5e-6",
    "Llama3.2-3B π1": "pi1-80pct_Q0-80pct-lr=5e-5",
    "Llama3.2-3B BoN(π1, Q1)": "BoN_pi1_Q1-60pct-lr=5e-6",
    "Llama3.2-3B π2": "pi2-60pct_Q1-60pct-lr=5e-6",
    "Llama3.2-3B BoN(π2, Q2)": "BoN_pi2_Q2-80pct-lr=5e-6",
    "Llama3.2-3B π3": "pi3-38pct_Q2-80pct-lr=5e-6"
}

# Create the inverse mapping from model_to_name_in_csv
name_to_model_in_csv = {v: k for k, v in model_to_name_in_csv.items()}

# Load the overall eval CSV
df = pd.read_csv(EVAL_CSV_PATH)

# Create the sub-df for the models to plot
df_to_plot = df[df['model'].isin(model_to_name_in_csv.values())]

# Change the model name to the key in model_to_name_in_csv
df_to_plot['model'] = df_to_plot['model'].map(name_to_model_in_csv)

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
colors = color_sequences['Dark2'] + color_sequences['Accent']

# Loop through metrics and categories
for row_idx, metric in enumerate(metrics):
    for col_idx, category in enumerate(categories):
        ax = axes[row_idx, col_idx]
        
        # Extract data
        avg_values = df_to_plot[f'{category} ({metric})']
        if metric == 'avg reward':
            avg_values = 20 + avg_values # Shift the reward to positive scale
        se_values = df_to_plot[f'{category} ({errors[row_idx]})']
        
        # Plot bar chart with error bars and colors for each model
        for i, (value, error) in enumerate(zip(avg_values, se_values)):
            ax.bar(x[i], value, yerr=error, capsize=5, alpha=0.7, width=bar_width, color=colors[i])
        
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
plt.savefig(f'playground/barplot/barplot_{current_time}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'playground/barplot/_barplot.png', dpi=300, bbox_inches='tight') # Have an accessible copy that stays at the top