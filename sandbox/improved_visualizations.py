import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data():
    """Load the merged data."""
    df = pd.read_csv("/home/INT/dienye.h/python_files/noise_test/dhcp_merged_data.csv")
    df['smoothing_iterations'] = pd.to_numeric(df['smoothing_iterations'])
    return df

def viz_option_1_boxplots(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 1: Box plots showing distributions at each smoothing level.
    Clearer view of data spread without overlapping error bars.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp', 'processing_time']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    n_hemispheres = df['hemisphere'].nunique()
    
    for idx, metric in enumerate(metrics):
        # Create box plot
        if n_hemispheres > 1:
            sns.boxplot(data=df, x='smoothing_iterations', y=metric, hue='hemisphere',
                       ax=axes[idx], palette=['#3498db', '#e74c3c'])
            axes[idx].legend(title='Hemisphere', fontsize=11)
        else:
            sns.boxplot(data=df, x='smoothing_iterations', y=metric,
                       ax=axes[idx], color='#3498db')
        
        axes[idx].set_xlabel('Smoothing Iterations', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=15, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'viz1_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/viz1_boxplots.png")
    plt.close()

def viz_option_2_percentage_change(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 2: Percentage change from baseline (0 iterations).
    Shows relative effects more clearly.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp', 'processing_time']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        for hemisphere in df['hemisphere'].unique():
            hemi_data = df[df['hemisphere'] == hemisphere]
            
            # Get baseline (0 iterations)
            baseline = hemi_data[hemi_data['smoothing_iterations'] == 0][metric].mean()
            
            # Calculate percentage change for each smoothing level
            summary = hemi_data.groupby('smoothing_iterations')[metric].mean()
            pct_change = ((summary - baseline) / baseline * 100)
            
            axes[idx].plot(pct_change.index, pct_change.values, 
                          marker='o', linewidth=3, markersize=10, 
                          label=hemisphere, alpha=0.8)
        
        axes[idx].axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        axes[idx].set_xlabel('Smoothing Iterations', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('% Change from Baseline', fontsize=13, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=15, fontweight='bold')
        axes[idx].legend(title='Hemisphere', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'viz2_percentage_change.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/viz2_percentage_change.png")
    plt.close()

def viz_option_3_normalized_comparison(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 3: All metrics normalized to 0-1 scale for direct comparison.
    Shows which metrics are most affected by smoothing.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp', 'processing_time']
    
    hemispheres = sorted(df['hemisphere'].unique())
    n_hemispheres = len(hemispheres)
    
    if n_hemispheres > 1:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        axes = [ax]
    
    for hemi_idx, hemisphere in enumerate(hemispheres):
        hemi_data = df[df['hemisphere'] == hemisphere]
        
        for metric in metrics:
            summary = hemi_data.groupby('smoothing_iterations')[metric].mean()
            
            # Normalize to 0-1 scale
            normalized = (summary - summary.min()) / (summary.max() - summary.min())
            
            axes[hemi_idx if n_hemispheres > 1 else 0].plot(
                normalized.index, normalized.values, 
                marker='o', linewidth=3, markersize=10, 
                label=metric.replace('_', ' ').title(), alpha=0.8
            )
        
        ax_idx = hemi_idx if n_hemispheres > 1 else 0
        axes[ax_idx].set_xlabel('Smoothing Iterations', fontsize=13, fontweight='bold')
        axes[ax_idx].set_ylabel('Normalized Value (0-1)', fontsize=13, fontweight='bold')
        title = f'{hemisphere.title()} Hemisphere' if n_hemispheres > 1 else 'Normalized Metric Comparison'
        axes[ax_idx].set_title(title, fontsize=15, fontweight='bold')
        axes[ax_idx].legend(fontsize=10, loc='best')
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'viz3_normalized_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/viz3_normalized_comparison.png")
    plt.close()

def viz_option_4_violin_plots(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 4: Violin plots showing full distributions.
    Best for seeing data density and outliers.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp', 'processing_time']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Check if we have multiple hemispheres
    n_hemispheres = df['hemisphere'].nunique()
    use_split = n_hemispheres == 2
    
    for idx, metric in enumerate(metrics):
        if n_hemispheres > 1:
            sns.violinplot(data=df, x='smoothing_iterations', y=metric, hue='hemisphere',
                          ax=axes[idx], palette=['#3498db', '#e74c3c'], 
                          split=use_split, inner='quartile')
        else:
            # Single hemisphere - no hue needed
            sns.violinplot(data=df, x='smoothing_iterations', y=metric,
                          ax=axes[idx], color='#3498db', inner='quartile')
        
        axes[idx].set_xlabel('Smoothing Iterations', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=15, fontweight='bold')
        if n_hemispheres > 1:
            axes[idx].legend(title='Hemisphere', fontsize=11)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'viz4_violin_plots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/viz4_violin_plots.png")
    plt.close()

def viz_option_5_individual_panels(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 5: Large individual panels for each metric with clean lines.
    Best for presentations or publications.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp', 'processing_time']
    colors = {'left': '#2E86AB', 'right': '#A23B72'}
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for hemisphere in df['hemisphere'].unique():
            hemi_data = df[df['hemisphere'] == hemisphere]
            summary = hemi_data.groupby('smoothing_iterations')[metric].agg(['mean', 'sem'])
            
            ax.plot(summary.index, summary['mean'], 
                   marker='o', linewidth=3, markersize=12, 
                   label=hemisphere.title(), color=colors.get(hemisphere, '#333333'),
                   alpha=0.9)
            
            # Add subtle shaded error region (SEM not SD for cleaner look)
            ax.fill_between(summary.index, 
                           summary['mean'] - summary['sem'],
                           summary['mean'] + summary['sem'],
                           alpha=0.2, color=colors.get(hemisphere, '#333333'))
        
        ax.set_xlabel('Smoothing Iterations', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} vs Smoothing Iterations', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(title='Hemisphere', fontsize=12, title_fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        safe_name = metric.replace('_', '-')
        plt.savefig(Path(output_dir) / f'viz5_{safe_name}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/viz5_{safe_name}.png")
        plt.close()

def viz_option_6_heatmap_summary(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 6: Heatmap showing all metrics and smoothing levels.
    Great for seeing patterns at a glance.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp']
    band_cols = [f'B{i}_power' for i in range(8)]
    all_metrics = metrics + band_cols
    
    # Calculate mean values for each smoothing level
    smoothing_levels = sorted(df['smoothing_iterations'].unique())
    
    # Create matrix of values (normalized)
    matrix_data = []
    metric_labels = []
    
    for metric in all_metrics:
        if metric in df.columns:
            row_data = []
            for level in smoothing_levels:
                mean_val = df[df['smoothing_iterations'] == level][metric].mean()
                row_data.append(mean_val)
            
            # Normalize each row to 0-1
            row_array = np.array(row_data)
            if row_array.max() != row_array.min():
                normalized = (row_array - row_array.min()) / (row_array.max() - row_array.min())
                matrix_data.append(normalized)
                metric_labels.append(metric.replace('_', ' ').title())
    
    matrix_data = np.array(matrix_data)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(matrix_data, annot=True, fmt='.2f', cmap='RdYlBu_r',
                xticklabels=[f'{int(x)}' for x in smoothing_levels],
                yticklabels=metric_labels, cbar_kws={'label': 'Normalized Value'},
                ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Smoothing Iterations', fontsize=13, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=13, fontweight='bold')
    ax.set_title('Normalized Metric Values Across Smoothing Levels', 
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'viz6_heatmap_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/viz6_heatmap_summary.png")
    plt.close()

def viz_option_7_bar_chart(df, output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Option 7: Bar charts with minimal error bars.
    Clean and easy to read for comparisons.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    metrics = ['total_power_above_b6', 'relative_power_above_b6', 'total_afp', 'processing_time']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    smoothing_levels = sorted(df['smoothing_iterations'].unique())
    x = np.arange(len(smoothing_levels))
    
    hemispheres = sorted(df['hemisphere'].unique())
    n_hemispheres = len(hemispheres)
    width = 0.35 if n_hemispheres > 1 else 0.6
    
    colors = {'left': '#3498db', 'right': '#e74c3c'}
    
    for idx, metric in enumerate(metrics):
        if n_hemispheres > 1:
            # Two hemispheres - side by side bars
            for hemi_idx, hemisphere in enumerate(hemispheres):
                means = []
                sems = []
                
                for level in smoothing_levels:
                    hemi_data = df[(df['smoothing_iterations'] == level) & 
                                  (df['hemisphere'] == hemisphere)][metric]
                    means.append(hemi_data.mean())
                    sems.append(hemi_data.sem())
                
                offset = (hemi_idx - 0.5) * width
                axes[idx].bar(x + offset, means, width, yerr=sems,
                            label=hemisphere.title(), 
                            color=colors.get(hemisphere, '#333333'), 
                            alpha=0.8, capsize=5)
            axes[idx].legend(title='Hemisphere', fontsize=11)
        else:
            # Single hemisphere - centered bars
            means = []
            sems = []
            hemisphere = hemispheres[0]
            
            for level in smoothing_levels:
                level_data = df[df['smoothing_iterations'] == level][metric]
                means.append(level_data.mean())
                sems.append(level_data.sem())
            
            axes[idx].bar(x, means, width, yerr=sems,
                        color=colors.get(hemisphere, '#3498db'), 
                        alpha=0.8, capsize=5, label=f'{hemisphere.title()} Hemisphere')
            axes[idx].legend(fontsize=11)
        
        axes[idx].set_xlabel('Smoothing Iterations', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=15, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([f'{int(level)}' for level in smoothing_levels])
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'viz7_bar_charts.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/viz7_bar_charts.png")
    plt.close()

def create_comparison_sheet(output_dir='/home/INT/dienye.h/python_files/noise_test/plots'):
    """
    Create a visual guide showing all 7 visualization options.
    """
    descriptions = {
        'viz1_boxplots.png': 'Box Plots: Shows distributions and outliers clearly',
        'viz2_percentage_change.png': 'Percentage Change: Shows relative effects from baseline',
        'viz3_normalized_comparison.png': 'Normalized: Direct comparison of different metrics',
        'viz4_violin_plots.png': 'Violin Plots: Shows full data density distributions',
        'viz5_*': 'Individual Panels: Large clean plots for presentations',
        'viz6_heatmap_summary.png': 'Heatmap: All metrics and smoothing levels at a glance',
        'viz7_bar_charts.png': 'Bar Charts: Clean comparisons with minimal error bars'
    }
    
    print("\n" + "="*80)
    print("VISUALIZATION OPTIONS CREATED")
    print("="*80)
    print("\nChoose the visualization style that works best for your needs:\n")
    
    for viz, desc in descriptions.items():
        print(f"  • {viz:35s} - {desc}")
    
    print("\nAll files saved in: ./plots/")

def main():
    print("="*80)
    print("GENERATING IMPROVED VISUALIZATIONS")
    print("="*80)
    
    df = load_data()
    print(f"\nLoaded data: {df.shape[0]} rows")
    print(f"Smoothing levels: {sorted(df['smoothing_iterations'].unique())}")
    print(f"Hemispheres: {df['hemisphere'].unique()}\n")
    
    print("Creating visualizations...\n")
    
    viz_option_1_boxplots(df)
    viz_option_2_percentage_change(df)
    viz_option_3_normalized_comparison(df)
    viz_option_4_violin_plots(df)
    viz_option_5_individual_panels(df)
    viz_option_6_heatmap_summary(df)
    viz_option_7_bar_chart(df)
    
    create_comparison_sheet()
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()