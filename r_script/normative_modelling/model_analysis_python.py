import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        print("Using default matplotlib style")

sns.set_palette("husl")

def load_model_results(csv_file_path):
    """
    Load the model comparison results from CSV file
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"File {csv_file_path} not found. Please check the path.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def explore_data(df):
    """
    Basic exploration of the model results
    """
    print("\n=== DATA EXPLORATION ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    if 'Y_feature' in df.columns:
        print(f"\nY variables analyzed: {df['Y_feature'].unique()}")
    if 'Site_Effect' in df.columns:
        print(f"Site effect types: {df['Site_Effect'].unique()}")
    if 'Complexity' in df.columns:
        print(f"Model complexities: {df['Complexity'].unique()}")
    
    print(f"\nAIC range: {df['AIC'].min():.2f} - {df['AIC'].max():.2f}")
    print(f"BIC range: {df['BIC'].min():.2f} - {df['BIC'].max():.2f}")

def find_best_models(df):
    """
    Find the best models for each Y variable based on AIC and BIC
    """
    print("\n=== BEST MODELS ===")
    
    best_models = {}
    
    for y_var in df['Y_feature'].unique():
        y_data = df[df['Y_feature'] == y_var].copy()
        
        # Best by AIC
        best_aic_idx = y_data['AIC'].idxmin()
        best_aic = y_data.loc[best_aic_idx]
        
        # Best by BIC  
        best_bic_idx = y_data['BIC'].idxmin()
        best_bic = y_data.loc[best_bic_idx]
        
        best_models[y_var] = {
            'best_aic': best_aic,
            'best_bic': best_bic
        }
        
        print(f"\n--- {y_var} ---")
        print(f"Best AIC: {best_aic['Model']} (AIC: {best_aic['AIC']:.2f})")
        if 'Site_Effect' in best_aic:
            print(f"  Site Effect: {best_aic['Site_Effect']}, Complexity: {best_aic['Complexity']}")
        
        print(f"Best BIC: {best_bic['Model']} (BIC: {best_bic['BIC']:.2f})")
        if 'Site_Effect' in best_bic:
            print(f"  Site Effect: {best_bic['Site_Effect']}, Complexity: {best_bic['Complexity']}")
    
    return best_models

def plot_model_comparison(df, save_plots=True, output_dir="model_plots"):
    """
    Create comprehensive plots for model comparison
    """
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. AIC vs BIC Scatter Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: AIC vs BIC for all models
    ax1 = axes[0, 0]
    if 'Y_feature' in df.columns:
        for y_var in df['Y_feature'].unique():
            y_data = df[df['Y_feature'] == y_var]
            ax1.scatter(y_data['AIC'], y_data['BIC'], label=y_var, alpha=0.7, s=60)
    else:
        ax1.scatter(df['AIC'], df['BIC'], alpha=0.7, s=60)
    
    ax1.set_xlabel('AIC', fontweight='bold')
    ax1.set_ylabel('BIC', fontweight='bold')
    ax1.set_title('AIC vs BIC for All Models')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Model complexity comparison
    ax2 = axes[0, 1]
    if 'Site_Effect' in df.columns and 'Complexity' in df.columns:
        # Create a combined complexity metric
        df_plot = df.copy()
        df_plot['Combined_Model'] = df_plot['Site_Effect'] + '_' + df_plot['Complexity']
        
        # Box plot of AIC by model type
        model_order = ['none_linear', 'none_smooth_mean', 'none_smooth_mean_var', 
                      'none_shash_constant', 'none_shash_full',
                      'fixed_linear', 'fixed_smooth_mean', 'fixed_smooth_mean_var',
                      'fixed_shash_constant', 'fixed_shash_full',
                      'random_linear', 'random_smooth_mean', 'random_smooth_mean_var',
                      'random_shash_constant', 'random_shash_full']
        
        # Filter to only existing model types
        existing_models = [m for m in model_order if m in df_plot['Combined_Model'].values]
        
        df_plot_filtered = df_plot[df_plot['Combined_Model'].isin(existing_models)]
        
        sns.boxplot(data=df_plot_filtered, x='Combined_Model', y='AIC', ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_title('AIC Distribution by Model Type')
        ax2.set_xlabel('Model Type', fontweight='bold')
        ax2.set_ylabel('AIC', fontweight='bold')
    else:
        # Fallback: simple bar plot
        model_aic = df.groupby('Model')['AIC'].mean().sort_values()
        ax2.bar(range(len(model_aic)), model_aic.values)
        ax2.set_xticks(range(len(model_aic)))
        ax2.set_xticklabels(model_aic.index, rotation=45, ha='right')
        ax2.set_title('Average AIC by Model')
        ax2.set_ylabel('Average AIC', fontweight='bold')
    
    # Plot 3: Delta AIC/BIC from best model
    ax3 = axes[1, 0]
    if 'Y_feature' in df.columns:
        delta_aic_data = []
        for y_var in df['Y_feature'].unique():
            y_data = df[df['Y_feature'] == y_var].copy()
            min_aic = y_data['AIC'].min()
            y_data['Delta_AIC'] = y_data['AIC'] - min_aic
            delta_aic_data.append(y_data)
        
        delta_df = pd.concat(delta_aic_data)
        
        if 'Site_Effect' in delta_df.columns:
            sns.boxplot(data=delta_df, x='Site_Effect', y='Delta_AIC', ax=ax3)
            ax3.set_title('Δ AIC by Site Effect Type')
        else:
            ax3.hist(delta_df['Delta_AIC'], bins=20, alpha=0.7)
            ax3.set_title('Distribution of Δ AIC')
        
        ax3.set_ylabel('Δ AIC from Best Model', fontweight='bold')
        ax3.set_xlabel('Site Effect Type', fontweight='bold')
    
    # Plot 4: Model performance heatmap
    ax4 = axes[1, 1]
    if 'Y_feature' in df.columns and 'Site_Effect' in df.columns and 'Complexity' in df.columns:
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(values='AIC', 
                                   index='Complexity', 
                                   columns='Site_Effect', 
                                   aggfunc='mean')
        
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', ax=ax4, fmt='.1f')
        ax4.set_title('Average AIC Heatmap')
        ax4.set_xlabel('Site Effect Type', fontweight='bold')
        ax4.set_ylabel('Model Complexity', fontweight='bold')
    else:
        # Alternative: correlation matrix of AIC and BIC
        corr_data = df[['AIC', 'BIC']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title('AIC-BIC Correlation')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{output_dir}/model_comparison_overview.png", dpi=300, bbox_inches='tight')
        print(f"Saved overview plot to {output_dir}/model_comparison_overview.png")
    
    plt.show()

def plot_detailed_analysis(df, save_plots=True, output_dir="model_plots"):
    """
    Create detailed analysis plots for each Y variable
    """
    if 'Y_feature' not in df.columns:
        print("No Y_feature column found. Skipping detailed analysis.")
        return
    
    for y_var in df['Y_feature'].unique():
        y_data = df[df['Y_feature'] == y_var].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Detailed Analysis for {y_var}', fontsize=16, fontweight='bold')
        
        # Plot 1: AIC Scores Bar Chart
        ax1 = axes[0, 0]
        if 'Site_Effect' in y_data.columns:
            # Group by site effect and create grouped bar chart
            site_effects = y_data['Site_Effect'].unique()
            x_pos = np.arange(len(y_data))
            
            # Create color map for site effects
            colors = {'none': 'skyblue', 'fixed': 'lightcoral', 'random': 'lightgreen'}
            
            bars = []
            for i, (_, row) in enumerate(y_data.iterrows()):
                color = colors.get(row['Site_Effect'], 'gray')
                bar = ax1.bar(i, row['AIC'], color=color, alpha=0.8, 
                             edgecolor='black', linewidth=0.5)
                bars.extend(bar)
        else:
            ax1.bar(range(len(y_data)), y_data['AIC'].values, color='skyblue', 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_title(f'AIC Scores - {y_var}', fontweight='bold')
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('AIC', fontweight='bold')
        ax1.set_xticks(range(len(y_data)))
        ax1.set_xticklabels(y_data['Model'].values, rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add legend for site effects
        if 'Site_Effect' in y_data.columns:
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors.get(se, 'gray'), 
                                           edgecolor='black', alpha=0.8) 
                             for se in site_effects]
            ax1.legend(legend_elements, site_effects, title='Site Effects', loc='upper right')
        
        # Plot 2: BIC Scores Bar Chart
        ax2 = axes[0, 1]
        if 'Site_Effect' in y_data.columns:
            for i, (_, row) in enumerate(y_data.iterrows()):
                color = colors.get(row['Site_Effect'], 'gray')
                ax2.bar(i, row['BIC'], color=color, alpha=0.8, 
                       edgecolor='black', linewidth=0.5)
        else:
            ax2.bar(range(len(y_data)), y_data['BIC'].values, color='lightcoral', 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_title(f'BIC Scores - {y_var}', fontweight='bold')
        ax2.set_xlabel('Models', fontweight='bold')
        ax2.set_ylabel('BIC', fontweight='bold')
        ax2.set_xticks(range(len(y_data)))
        ax2.set_xticklabels(y_data['Model'].values, rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legend for site effects
        if 'Site_Effect' in y_data.columns:
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors.get(se, 'gray'), 
                                           edgecolor='black', alpha=0.8) 
                             for se in site_effects]
            ax2.legend(legend_elements, site_effects, title='Site Effects', loc='upper right')
        
        # Plot 3: Percentage Difference in Performance Bar Chart (Based on BIC)
        ax3 = axes[0, 2]
        min_bic = y_data['BIC'].min()
        
        y_data['BIC_Pct_Diff'] = ((y_data['BIC'] - min_bic) / min_bic) * 100
        
        # Create bar chart for BIC percentage differences
        bars = ax3.bar(range(len(y_data)), y_data['BIC_Pct_Diff'], 
                       color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Color bars based on site effect type if available
        if 'Site_Effect' in y_data.columns:
            colors = {'none': 'skyblue', 'fixed': 'lightcoral', 'random': 'lightgreen'}
            for i, (_, row) in enumerate(y_data.iterrows()):
                bars[i].set_color(colors.get(row['Site_Effect'], 'gray'))
        
        ax3.set_title(f'BIC Performance Difference from Best Model - {y_var}', fontweight='bold')
        ax3.set_xlabel('Models', fontweight='bold')
        ax3.set_ylabel('% Difference from Best Model (BIC)', fontweight='bold')
        ax3.set_xticks(range(len(y_data)))
        ax3.set_xticklabels(y_data['Model'].values, rotation=45, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        
        # Add legend for site effects
        if 'Site_Effect' in y_data.columns:
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors.get(se, 'gray'), 
                                           edgecolor='black', alpha=0.8) 
                             for se in y_data['Site_Effect'].unique()]
            ax3.legend(legend_elements, y_data['Site_Effect'].unique(), 
                      title='Site Effects', loc='upper right')
        
        # Add value labels on bars for significant differences
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.5:  # Only label if difference is substantial
                ax3.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
        
        # Plot 4: AIC Comparison - ALL MODELS with Color Coding
        ax4 = axes[1, 0]
        
        # Find the best model overall (lowest AIC)
        best_aic_model = y_data.loc[y_data['AIC'].idxmin()]
        
        # Color scheme for site effects
        site_colors = {'none': 'blue', 'fixed': 'orange', 'random': 'green'}
        
        if 'Site_Effect' in y_data.columns:
            # Plot all site effects across all model sets
            for site_effect in ['none', 'fixed', 'random']:
                if site_effect in y_data['Site_Effect'].values:
                    site_data = y_data[y_data['Site_Effect'] == site_effect].copy()
                    
                    # Sort models logically: m1a, m1b, m1c, m1d, m1e, m2a, m2b, etc.
                    site_data['sort_key1'] = site_data['Model'].str.extract(r'm([123])')[0].astype(int)  # Extract model set number
                    site_data['sort_key2'] = site_data['Model'].str.extract(r'([a-e])')[0]  # Extract complexity letter
                    site_data = site_data.sort_values(['sort_key1', 'sort_key2'])
                    
                    # Create x-positions (0-14 for 15 models)
                    x_positions = range(len(site_data))
                    
                    # Label with model set information
                    if site_effect == 'none':
                        label = 'no site effects (m1)'
                    elif site_effect == 'fixed':
                        label = 'fixed site effects (m2)'
                    elif site_effect == 'random':
                        label = 'random site effects (m3)'
                    
                    # Add BEST marker to winning approach
                    if site_effect == best_aic_model['Site_Effect']:
                        label += ' - BEST'
                    
                    ax4.plot(x_positions, site_data['AIC'].values, 
                            marker='o', label=label, 
                            linewidth=2, markersize=6, color=site_colors[site_effect])
            
            # Mark the specific best model with a gold star
            # Find position of best model in the sorted data
            best_site_data = y_data[y_data['Site_Effect'] == best_aic_model['Site_Effect']].copy()
            best_site_data['sort_key1'] = best_site_data['Model'].str.extract(r'm([123])')[0].astype(int)
            best_site_data['sort_key2'] = best_site_data['Model'].str.extract(r'([a-e])')[0]
            best_site_data = best_site_data.sort_values(['sort_key1', 'sort_key2'])
            
            best_model_pos = None
            for i, (_, row) in enumerate(best_site_data.iterrows()):
                if row['Model'] == best_aic_model['Model']:
                    best_model_pos = i
                    break
            
            if best_model_pos is not None:
                ax4.scatter(best_model_pos, best_aic_model['AIC'], 
                           s=300, color='gold', edgecolors='black', linewidth=2, 
                           marker='*', zorder=10, label='Best AIC Model')
            
            # Create simplified x-axis labels: just a, b, c, d, e (one set only)
            aic_model_labels = ['a', 'b', 'c', 'd', 'e']  # 5 labels for 15 models
            
        else:
            aic_model_labels = ['a', 'b', 'c', 'd', 'e']
        
        ax4.set_title(f'AIC Evolution Across All Models - {y_var}', fontweight='bold')
        ax4.set_xlabel('Model Progression', fontweight='bold')
        ax4.set_ylabel('AIC', fontweight='bold')
        ax4.set_xticks(range(len(aic_model_labels)))
        ax4.set_xticklabels(aic_model_labels, rotation=0, ha='center', fontsize=10)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: BIC Comparison - ALL MODELS with Color Coding
        ax5 = axes[1, 1]
        
        # Find the best model overall (lowest BIC)
        best_bic_model = y_data.loc[y_data['BIC'].idxmin()]
        
        if 'Site_Effect' in y_data.columns:
            # Plot all site effects across all model sets
            for site_effect in ['none', 'fixed', 'random']:
                if site_effect in y_data['Site_Effect'].values:
                    site_data = y_data[y_data['Site_Effect'] == site_effect].copy()
                    
                    # Sort models logically
                    site_data['sort_key1'] = site_data['Model'].str.extract(r'm([123])')[0].astype(int)
                    site_data['sort_key2'] = site_data['Model'].str.extract(r'([a-e])')[0]
                    site_data = site_data.sort_values(['sort_key1', 'sort_key2'])
                    
                    # Create x-positions
                    x_positions = range(len(site_data))
                    
                    # Label with model set information
                    if site_effect == 'none':
                        label = 'no site effects (m1)'
                    elif site_effect == 'fixed':
                        label = 'fixed site effects (m2)'
                    elif site_effect == 'random':
                        label = 'random site effects (m3)'
                    
                    # Add BEST marker to winning approach
                    if site_effect == best_bic_model['Site_Effect']:
                        label += ' - BEST'
                    
                    ax5.plot(x_positions, site_data['BIC'].values, 
                            marker='s', label=label, 
                            linewidth=2, markersize=6, color=site_colors[site_effect])
            
            # Mark the specific best model with a gold star
            best_bic_site_data = y_data[y_data['Site_Effect'] == best_bic_model['Site_Effect']].copy()
            best_bic_site_data['sort_key1'] = best_bic_site_data['Model'].str.extract(r'm([123])')[0].astype(int)
            best_bic_site_data['sort_key2'] = best_bic_site_data['Model'].str.extract(r'([a-e])')[0]
            best_bic_site_data = best_bic_site_data.sort_values(['sort_key1', 'sort_key2'])
            
            best_bic_model_pos = None
            for i, (_, row) in enumerate(best_bic_site_data.iterrows()):
                if row['Model'] == best_bic_model['Model']:
                    best_bic_model_pos = i
                    break
            
            if best_bic_model_pos is not None:
                ax5.scatter(best_bic_model_pos, best_bic_model['BIC'], 
                           s=300, color='gold', edgecolors='black', linewidth=2, 
                           marker='*', zorder=10, label='Best BIC Model')
            
            # Create simplified x-axis labels: just a, b, c, d, e (one set only)
            bic_model_labels = ['a', 'b', 'c', 'd', 'e']  # 5 labels for 15 models
        
        else:
            bic_model_labels = ['a', 'b', 'c', 'd', 'e']
        
        ax5.set_title(f'BIC Evolution Across All Models - {y_var}', fontweight='bold')
        ax5.set_xlabel('Model Progression', fontweight='bold')
        ax5.set_ylabel('BIC', fontweight='bold')
        ax5.set_xticks(range(len(bic_model_labels)))
        ax5.set_xticklabels(bic_model_labels, rotation=0, ha='center', fontsize=10)
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Best Model Summary
        ax6 = axes[1, 2]
        ax6.axis('off')  # Turn off axis for text display
        
        # Find best models
        best_aic_idx = y_data['AIC'].idxmin()
        best_bic_idx = y_data['BIC'].idxmin()
        best_aic_model = y_data.loc[best_aic_idx]
        best_bic_model = y_data.loc[best_bic_idx]
        
        # Create summary text
        summary_text = f"BEST MODELS FOR {y_var}\n\n"
        summary_text += f"Best AIC:\n"
        summary_text += f"  Model: {best_aic_model['Model']}\n"
        summary_text += f"  AIC: {best_aic_model['AIC']:.2f}\n"
        if 'Site_Effect' in best_aic_model:
            summary_text += f"  Site Effect: {best_aic_model['Site_Effect']}\n"
            summary_text += f"  Complexity: {best_aic_model['Complexity']}\n"
        
        summary_text += f"\nBest BIC:\n"
        summary_text += f"  Model: {best_bic_model['Model']}\n"
        summary_text += f"  BIC: {best_bic_model['BIC']:.2f}\n"
        if 'Site_Effect' in best_bic_model:
            summary_text += f"  Site Effect: {best_bic_model['Site_Effect']}\n"
            summary_text += f"  Complexity: {best_bic_model['Complexity']}\n"
        
        # Add performance statistics
        summary_text += f"\nPERFORMANCE STATS:\n"
        summary_text += f"AIC Range: {y_data['AIC'].min():.1f} - {y_data['AIC'].max():.1f}\n"
        summary_text += f"BIC Range: {y_data['BIC'].min():.1f} - {y_data['BIC'].max():.1f}\n"
        summary_text += f"Models Tested: {len(y_data)}\n"
        
        # Check if AIC and BIC agree
        if best_aic_model['Model'] == best_bic_model['Model']:
            summary_text += f"\n✓ AIC and BIC AGREE\n"
            summary_text += f"RECOMMENDED: {best_aic_model['Model']}"
        else:
            summary_text += f"\n⚠ AIC and BIC DISAGREE\n"
            summary_text += f"Consider BIC for parsimony"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        if save_plots:
            safe_y_var = y_var.replace('/', '_').replace(' ', '_')
            plt.savefig(f"{output_dir}/detailed_analysis_{safe_y_var}.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Saved detailed plot for {y_var}")
        
        plt.show()

def generate_summary_report(df, best_models):
    """
    Generate a comprehensive summary report
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL ANALYSIS REPORT")
    print("="*60)
    
    # Overall statistics
    print(f"\nTOTAL MODELS ANALYZED: {len(df)}")
    if 'Y_feature' in df.columns:
        print(f"Y VARIABLES: {len(df['Y_feature'].unique())}")
        for y_var in df['Y_feature'].unique():
            y_count = len(df[df['Y_feature'] == y_var])
            print(f"  - {y_var}: {y_count} models")
    
    # Model performance summary
    print(f"\nMODEL PERFORMANCE SUMMARY:")
    print(f"AIC Range: {df['AIC'].min():.2f} to {df['AIC'].max():.2f}")
    print(f"BIC Range: {df['BIC'].min():.2f} to {df['BIC'].max():.2f}")
    print(f"Mean AIC: {df['AIC'].mean():.2f} (±{df['AIC'].std():.2f})")
    print(f"Mean BIC: {df['BIC'].mean():.2f} (±{df['BIC'].std():.2f})")
    
    # Site effect analysis
    if 'Site_Effect' in df.columns:
        print(f"\nSITE EFFECT ANALYSIS:")
        site_performance = df.groupby('Site_Effect')[['AIC', 'BIC']].agg(['mean', 'std', 'min'])
        for site_effect in df['Site_Effect'].unique():
            site_data = df[df['Site_Effect'] == site_effect]
            print(f"\n{site_effect.upper()} site effects:")
            print(f"  Models: {len(site_data)}")
            print(f"  AIC: {site_data['AIC'].mean():.2f} ± {site_data['AIC'].std():.2f}")
            print(f"  BIC: {site_data['BIC'].mean():.2f} ± {site_data['BIC'].std():.2f}")
            print(f"  Best AIC: {site_data['AIC'].min():.2f}")
            print(f"  Best BIC: {site_data['BIC'].min():.2f}")
    
    # Complexity analysis
    if 'Complexity' in df.columns:
        print(f"\nCOMPLEXITY ANALYSIS:")
        for complexity in df['Complexity'].unique():
            comp_data = df[df['Complexity'] == complexity]
            print(f"\n{complexity.upper()}:")
            print(f"  Models: {len(comp_data)}")
            print(f"  AIC: {comp_data['AIC'].mean():.2f} ± {comp_data['AIC'].std():.2f}")
            print(f"  BIC: {comp_data['BIC'].mean():.2f} ± {comp_data['BIC'].std():.2f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if 'Y_feature' in df.columns:
        for y_var, models in best_models.items():
            print(f"\nFor {y_var}:")
            if models['best_aic']['Model'] == models['best_bic']['Model']:
                print(f"  CONSISTENT WINNER: {models['best_aic']['Model']}")
                if 'Site_Effect' in models['best_aic']:
                    print(f"    Site effects: {models['best_aic']['Site_Effect']}")
                    print(f"    Complexity: {models['best_aic']['Complexity']}")
            else:
                print(f"  AIC recommends: {models['best_aic']['Model']}")
                print(f"  BIC recommends: {models['best_bic']['Model']}")
                print(f"  Consider: BIC is more conservative, prefer it for parsimony")

def main():
    """
    Main function to run the complete analysis
    """
    # File path - modify this to match your CSV file location
    csv_file_path = "comprehensive_model_comparison_results.csv"
    # Alternative common names:
    # csv_file_path = "model_comparison_results.csv"
    # csv_file_path = "model_comparison_results_random_site_effects.csv"
    
    # Load data
    df = load_model_results(csv_file_path)
    if df is None:
        return
    
    # Explore data
    explore_data(df)
    
    # Find best models
    best_models = find_best_models(df)
    
    # Create plots
    print("\nCreating overview plots...")
    plot_model_comparison(df, save_plots=True)
    
    print("\nCreating detailed plots...")
    plot_detailed_analysis(df, save_plots=True)
    
    # Generate summary report
    generate_summary_report(df, best_models)
    
    print(f"\nAnalysis complete! Check the 'model_plots' directory for saved plots.")

if __name__ == "__main__":
    main()