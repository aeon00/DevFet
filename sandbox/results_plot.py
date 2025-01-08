import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("sandbox/combined_results.csv")
max_gest_age_val = max(df['gestational_age'])

print(df[df['gestational_age']==max_gest_age_val])


# Set the style
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create the line plot
sns.lineplot(data=df,
            x='gestational_age',     # Replace with your x-axis column
            y='analyze_folding_power',    # Replace with your y-axis column
            linewidth=2.5,
            marker='o',           # Add markers at data points
            markersize=8)

# Customize the appearance
plt.title('Line Graph', pad=15, fontsize=14)
plt.xlabel('X Label', labelpad=10)
plt.ylabel('Y Label', labelpad=10)

# Enhance the visuals
sns.despine(left=False, bottom=False)  # Keep the axes
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


# # Set figure size and style
# plt.figure(figsize=(12, 6))
# sns.set_style("whitegrid")

# # Plot all three lines
# sns.lineplot(data=df, x='volume_ml', y='band_power_B4', label='B4', marker='o')
# sns.lineplot(data=df, x='volume_ml', y='band_power_B5', label='B5', marker='s')
# sns.lineplot(data=df, x='volume_ml', y='band_power_B6', label='B6', marker='^')

# # Customize the appearance
# plt.title('Multiple Variables Over Time', pad=15, fontsize=14)
# plt.xlabel('X Label', labelpad=10)
# plt.ylabel('Y Values', labelpad=10)

# # Enhance readability
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, linestyle='--', alpha=0.7)

# plt.tight_layout()
# plt.show()

# # Set the style
# sns.set_style("whitegrid")
# plt.figure(figsize=(12, 7))

# # Create histogram
# sns.histplot(
#     data=df,
#     x='gestational_age',  # Match the DataFrame column name
#     bins=30,
#     color='#2E86C1',        # Slightly darker blue for better visibility
#     alpha=0.8,
#     kde=True,
#     line_kws={'color': '#E74C3C', 'linewidth': 2}
# )

# # Add title and labels
# plt.title('Gestational Across Subjects', fontsize=14, pad=15)
# plt.xlabel('Gestational Age', fontsize=12)
# plt.ylabel('Number of subjects', fontsize=12)

# # Add grid for better readability
# plt.grid(True, alpha=0.3)

# # Format x-axis with thousand separators
# plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# # Adjust layout
# plt.tight_layout()

# # Show the plot
# plt.show()