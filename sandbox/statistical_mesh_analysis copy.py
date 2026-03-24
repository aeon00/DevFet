import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import glob

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def read_and_parse_csv(file_path):
    """
    Read CSV file and parse dictionary data from each cell.
    Returns a list of dictionaries.
    """
    df = pd.read_csv(file_path)
    
    all_records = []
    
    # Iterate through each column (0, 1, 2, 3)
    for col in df.columns:
        # Get the data row (skip header row)
        for idx, cell_data in df[col].items():
            if pd.notna(cell_data):
                try:
                    # Parse the dictionary string
                    if isinstance(cell_data, str):
                        record = ast.literal_eval(cell_data)
                    else:
                        record = cell_data
                    
                    # Add source information
                    record['source_file'] = Path(file_path).name
                    record['source_column'] = col
                    all_records.append(record)
                except:
                    print(f"Error parsing data in {file_path}, column {col}, row {idx}")
                    continue
    
    return all_records

def merge_all_csvs(directory_path):
    """
    Merge all CSV files from a directory into one DataFrame.
    """
    csv_files = glob.glob(str(Path(directory_path) / "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {Path(f).name}")
    
    all_data = []
    
    for csv_file in csv_files:
        records = read_and_parse_csv(csv_file)
        all_data.extend(records)
        print(f"Loaded {len(records)} records from {Path(csv_file).name}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(f"\nTotal records loaded: {len(df)}")
    
    return df

def analyze_smoothing_effects(df):
    """
    Analyze the effect of smoothing iterations on various metrics.
    """
    # Ensure smoothing_iterations is numeric
    df['smoothing_iterations'] = pd.to_numeric(df['smoothing_iterations'])
    
    # Identify power and other numeric columns
    power_cols = [col for col in df.columns if 'power' in col.lower()]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove identification columns from analysis
    exclude_cols = ['smoothing_iterations', 'n_bands', 'gestational_age']
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print("\n" + "="*80)
    print("SMOOTHING ITERATION EFFECTS ANALYSIS")
    print("="*80)
    
    # 1. Summary statistics by smoothing iteration
    print("\n1. Summary Statistics by Smoothing Iteration:")
    print("-" * 80)
    
    for col in analysis_cols[:10]:  # Show first 10 metrics
        print(f"\n{col}:")
        summary = df.groupby('smoothing_iterations')[col].agg(['mean', 'std', 'min', 'max'])
        print(summary)
    
    # 2. Correlation analysis
    print("\n2. Correlation between Smoothing Iterations and Metrics:")
    print("-" * 80)
    
    correlations = {}
    for col in analysis_cols:
        if col in df.columns:
            corr = df['smoothing_iterations'].corr(df[col])
            correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for metric, corr in sorted_corr:
        print(f"{metric:40s}: {corr:7.4f}")
    
    return analysis_cols, correlations

def create_visualizations(df, analysis_cols, output_dir='./plots'):
    """
    Create comprehensive visualizations of smoothing effects.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. Line plots showing trends across smoothing iterations
    print("\nGenerating visualizations...")
    
    # Plot top metrics affected by smoothing
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metrics_to_plot = ['total_power_above_b6', 'relative_power_above_b6', 
                       'total_afp', 'processing_time']
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            # Group by smoothing iterations and hemisphere
            grouped = df.groupby(['smoothing_iterations', 'hemisphere'])[metric].agg(['mean', 'std'])
            
            for hemisphere in df['hemisphere'].unique():
                data = grouped.xs(hemisphere, level='hemisphere')
                axes[idx].errorbar(data.index, data['mean'], yerr=data['std'], 
                                 label=hemisphere, marker='o', capsize=5)
            
            axes[idx].set_xlabel('Smoothing Iterations')
            axes[idx].set_ylabel(metric)
            axes[idx].set_title(f'{metric} vs Smoothing Iterations')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'smoothing_effects_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/smoothing_effects_overview.png")
    plt.close()
    
    # 2. Band power comparison across smoothing levels
    band_cols = [col for col in df.columns if col.startswith('B') and 'power' in col]
    
    if band_cols:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, band_col in enumerate(band_cols):
            if idx < len(axes):
                grouped = df.groupby(['smoothing_iterations', 'hemisphere'])[band_col].mean().unstack()
                grouped.plot(ax=axes[idx], marker='o')
                axes[idx].set_xlabel('Smoothing Iterations')
                axes[idx].set_ylabel('Power')
                axes[idx].set_title(f'{band_col}')
                axes[idx].legend(title='Hemisphere')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'band_power_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/band_power_analysis.png")
        plt.close()
    
    # 3. Heatmap of correlations
    # Calculate correlation matrix for different smoothing levels
    smoothing_levels = sorted(df['smoothing_iterations'].unique())
    
    fig, axes = plt.subplots(1, len(smoothing_levels), figsize=(20, 5))
    if len(smoothing_levels) == 1:
        axes = [axes]
    
    for idx, smooth_level in enumerate(smoothing_levels):
        subset = df[df['smoothing_iterations'] == smooth_level]
        numeric_subset = subset[analysis_cols[:8]].select_dtypes(include=[np.number])
        
        corr_matrix = numeric_subset.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=axes[idx], cbar_kws={'label': 'Correlation'})
        axes[idx].set_title(f'Smoothing Iterations = {smooth_level}')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/correlation_heatmaps.png")
    plt.close()
    
    # 4. Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in df.columns and idx < len(axes):
            for smooth_level in sorted(df['smoothing_iterations'].unique()):
                subset = df[df['smoothing_iterations'] == smooth_level][metric].dropna()
                axes[idx].hist(subset, alpha=0.5, label=f'Smooth={smooth_level}', bins=20)
            
            axes[idx].set_xlabel(metric)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {metric}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/distribution_comparison.png")
    plt.close()

def compare_across_datasets(df):
    """
    Compare metrics across different source files (datasets).
    """
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)
    
    # Summary by source file and smoothing iteration
    print("\n1. Mean values by dataset and smoothing iteration:")
    print("-" * 80)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp']
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\n{metric}:")
            pivot = df.pivot_table(values=metric, 
                                  index='smoothing_iterations', 
                                  columns='source_file', 
                                  aggfunc='mean')
            print(pivot)
    
    # Statistical comparison
    print("\n2. Coefficient of Variation (CV) across datasets:")
    print("-" * 80)
    
    for metric in metrics:
        if metric in df.columns:
            cv = df.groupby(['smoothing_iterations', 'source_file'])[metric].agg(['mean', 'std'])
            cv['cv'] = (cv['std'] / cv['mean']) * 100
            print(f"\n{metric}:")
            print(cv['cv'].unstack())

def main():
    """
    Main execution function.
    """
    # CHANGE THIS to your directory path
    directory_path = "/home/INT/dienye.h/python_files/noise_test"  # Update this path
    
    # Step 1: Merge all CSV files
    print("="*80)
    print("STEP 1: MERGING CSV FILES")
    print("="*80)
    
    df_merged = merge_all_csvs(directory_path)
    
    # Save merged data
    output_file = Path(directory_path) / "dhcp_merged_data.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"\nMerged data saved to: {output_file}")
    
    # Display basic info
    print("\n" + "="*80)
    print("MERGED DATA OVERVIEW")
    print("="*80)
    print(f"\nShape: {df_merged.shape}")
    print(f"\nColumns: {df_merged.columns.tolist()}")
    print(f"\nSmoothing iterations present: {sorted(df_merged['smoothing_iterations'].unique())}")
    print(f"\nHemispheres: {df_merged['hemisphere'].unique()}")
    print(f"\nGestational age range: {df_merged['gestational_age'].min():.2f} - {df_merged['gestational_age'].max():.2f}")
    
    print("\nFirst few rows:")
    print(df_merged.head())
    
    # Step 2: Analyze smoothing effects
    analysis_cols, correlations = analyze_smoothing_effects(df_merged)
    
    # Step 3: Create visualizations
    print("\n" + "="*80)
    print("STEP 2: CREATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(df_merged, analysis_cols)
    
    # Step 4: Cross-dataset comparison
    if df_merged['source_file'].nunique() > 1:
        compare_across_datasets(df_merged)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nMerged data: merged_data.csv")
    print(f"Visualizations: ./plots/ directory")
    
    return df_merged

if __name__ == "__main__":
    df = main()