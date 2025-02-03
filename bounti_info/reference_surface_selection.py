import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/envau/work/meca/users/dienye.h/bounti_analysis/vertices_surface_area_and_volume.csv')

# Calculate 95th percentile for each numeric column
percentile_95 = df[['Number of Vertices', 'Surface Area Values', 'Volume Values']].quantile(0.95)

# Function to find files corresponding to values closest to 95th percentile
def find_closest_file(df, column, target_value):
    # Find the row with value closest to the 95th percentile
    closest_idx = (df[column] - target_value).abs().idxmin()
    return {
        'File': df.loc[closest_idx, 'File name'],
        f'{column}': df.loc[closest_idx, column],
        'Percentile_Value': target_value
    }

# Find files for each metric
results = {}
for column in ['Number of Vertices', 'Surface Area Values', 'Volume Values']:
    results[column] = find_closest_file(df, column, percentile_95[column])

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Metric': list(results.keys()),
    'File': [results[k]['File'] for k in results],
    'Actual_Value': [results[k][k] for k in results],
    'Percentile_Value': [results[k]['Percentile_Value'] for k in results]
})

# Print results
print("\n95th Percentile Analysis:")
print("------------------------")
for metric in results:
    print(f"\n{metric}:")
    print(f"File: {results[metric]['File']}")
    print(f"Value: {results[metric][metric]:.2f}")
    print(f"95th Percentile Value: {results[metric]['Percentile_Value']:.2f}")

# Save results to CSV
summary_df.to_csv('/envau/work/meca/users/dienye.h/bounti_analysis/95th_percentile_analysis.csv', index=False)