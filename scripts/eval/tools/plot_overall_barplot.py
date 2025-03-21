import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import color_sequences

REWARD_MIN = -20
REWARD_MAX = 0
SUCCESS_RATE_MIN = 0
SUCCESS_RATE_MAX = 1

# Load the CSV file
df = pd.read_csv('/share/portal/hw575/agent_prm/playground/barplot/barplot.csv')  # Replace 'your_file.csv' with your actual file path

# Define columns for plotting
categories = ['train', 'val', 'test', 'total']
metrics = ['avg reward', 'avg success rate']
errors = ['se reward', 'se success rate']

# Number of models
models = df['model']
x = np.arange(len(models))  # The label locations

# Create the figure and axes
fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)

# Define bar width
bar_width = 0.6

# Define color sequence
colors = color_sequences['Dark2']

# Loop through metrics and categories
for row_idx, metric in enumerate(metrics):
    for col_idx, category in enumerate(categories):
        ax = axes[row_idx, col_idx]
        
        # Extract data
        avg_values = df[f'{category} ({metric})']
        se_values = df[f'{category} ({errors[row_idx]})']
        
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

# Save the plot to a file
plt.savefig('playground/barplot/barplot.png', dpi=300, bbox_inches='tight')
