import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_band_powers_seaborn(csv_path, output_dir=None):
    """
    Create seaborn plots showing band power changes with smoothing iterations.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the analysis results
    output_dir : str, optional
        Directory to save the plots. If None, plots will be displayed
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Reshape data for seaborn
    melted_df = pd.melt(df, 
                        id_vars=['N', 'Number of smoothing iterations'],
                        value_vars=['band_power_B4', 'band_power_B5', 'band_power_B6'],
                        var_name='Band',
                        value_name='Power')
    
    # Clean up band names
    melted_df['Band'] = melted_df['Band'].str.replace('band_power_', '')
    
    # Set up the style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Create figure for line plots
    plt.figure(figsize=(15, 10))
    
    # Create line plot with seaborn
    g = sns.relplot(data=melted_df,
                    x='Number of smoothing iterations',
                    y='Power',
                    hue='Band',
                    col='N',
                    kind='line',
                    marker='o',
                    height=3,
                    aspect=1.2,
                    col_wrap=3,
                    facet_kws={'sharex': False, 'sharey': False})
    
    # Customize the plot
    g.fig.suptitle('Band Power Changes with Smoothing Iterations', y=1.02, fontsize=16)
    g.set_titles("N = {col_name}")
    g.set_axis_labels("Smoothing Iterations", "Band Power")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_dir:
        output_path = f"{output_dir}/band_power_analysis_seaborn.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Line plots saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Create heatmaps
    plt.figure(figsize=(18, 6))
    
    # Create a heatmap for each band
    for idx, band in enumerate(['B4', 'B5', 'B6']):
        plt.subplot(1, 3, idx + 1)
        
        # Create pivot table for heatmap
        pivot_data = df.pivot(
            index='N',
            columns='Number of smoothing iterations',
            values=f'band_power_{band}'
        )
        
        # Plot heatmap
        sns.heatmap(pivot_data,
                    cmap='viridis',
                    cbar_kws={'label': 'Band Power'},
                    annot=True,  # Show values in cells
                    fmt='.2f',   # Format for annotations
                    )
        
        plt.title(f'Band {band} Power')
        plt.xlabel('Smoothing Iterations')
        plt.ylabel('N Value')
    
    plt.suptitle('Band Power Heatmaps', y=1.05, fontsize=16)
    plt.tight_layout()
    
    # Save or display heatmaps
    if output_dir:
        output_path = f"{output_dir}/band_power_heatmaps_seaborn.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

# Example usage
if __name__ == "__main__":
    csv_path = "/home/INT/dienye.h/Python Codes/QA_sub-0001_ses-0001_results.csv"  # Update this path
    output_dir = None  # Update this path or set to None to display plots
    
    plot_band_powers_seaborn(csv_path, output_dir)