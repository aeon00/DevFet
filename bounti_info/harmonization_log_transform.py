import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from neuroHarmonize import harmonizationLearn

# ==========================================
# 1. FILE PATHS & SETUP
# ==========================================
input_file = "/home/INT/dienye.h/python_files/combined_dataset/all_sites_double_filt_combined.csv"
output_dir = "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref"
plot_dir = "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/plots/by_cohort/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True) # Ensure plot directory exists

data_df = pd.read_csv(input_file)

# Setup variables
data_df['batch'] = data_df['SITE']  
data_df['gestational_age'] = pd.to_numeric(data_df['gestational_age'], errors='coerce')

brain_cols = [
    'analyze_folding_power', 'surface_area_cm2', 'B4_vertex_percentage', 'B5_vertex_percentage', 'B6_vertex_percentage',
    'band_parcels_B4', 'band_parcels_B5', 'band_parcels_B6', 'volume_ml', 'gyrification_index', 'hull_area', 'B4_surface_area',
    'B5_surface_area', 'B6_surface_area', 'B4_surface_area_percentage', 'B5_surface_area_percentage', 'B6_surface_area_percentage',
    'band_power_B4', 'band_power_B5', 'band_power_B6', 'B4_band_relative_power', 'B5_band_relative_power', 'B6_band_relative_power'
]

covars = pd.DataFrame({
    "age": data_df["gestational_age"],
    "batch": pd.Categorical(data_df["batch"]).codes
})

# ==========================================
# 2. LOG TRANSFORM & SCALING (The Fix)
# ==========================================
print("Applying log transformation and scaling...")

# A. Log-transform (safely handles zeros)
data_log = np.log1p(data_df[brain_cols])



# ==========================================
# 3. HARMONIZATION
# ==========================================
print("Running neuroHarmonize (this may take a few minutes)...")
covars_harm = covars.copy()
covars_harm['SITE'] = covars_harm['batch'].astype(str)
covars_harm = covars_harm.drop(columns=['batch'])

# A. Harmonize directly on the log-transformed data
estimates, harmonized_data_log, s_data = harmonizationLearn(
    data_log.values, # .values prevents pandas column name warnings
    covars=covars_harm, 
    smooth_terms=["age"], 
    return_s_data=True,
    ref_batch="2",
    eb=True # Removed ref_batch to harmonize to the grand mean
)

# ==========================================
# 4. REVERSE LOG
# ==========================================
print("Reversing transformations...")

# A. Back-transform to the native biological scale
harmonized_final_data = np.expm1(harmonized_data_log)


# Build the final dataframe
harmonized_df = data_df.copy()
rescaled_harmonized_df = pd.DataFrame(harmonized_final_data, columns=brain_cols)
harmonized_df = pd.concat([harmonized_df, rescaled_harmonized_df.add_suffix("_harm")], axis=1)

# Save the final data
harmonized_df.to_csv(os.path.join(output_dir, "harmonized_log_trans_all_sites.csv"), index=False)
print("Harmonization complete. Generating plots...")


# ==========================================
# 4.5 EXTRACT & PLOT QUANTIFIABLE SITE EFFECTS
# ==========================================
print("Quantifying and saving site effects...")

# 1. Get the unique site names in the exact order ComBat processed them
site_names = np.unique(covars_harm['SITE'])

# 2. Extract the additive batch effects (gamma_star) from the model estimates
# This is the amount the model shifted the data in the LOG space
gamma_star = estimates['gamma_star']

# 3. Create a DataFrame for the raw log-shifts
gamma_df = pd.DataFrame(gamma_star, columns=brain_cols, index=site_names)
gamma_df.index.name = 'Site'

# 4. Convert log-shifts to highly interpretable PERCENTAGE shifts
# Math: (exp(gamma) - 1) * 100 calculates the % inflation/deflation caused by the scanner
pct_shift_df = (np.expm1(gamma_df)) * 100

# Save both to CSV so you have the raw numbers for your paper/tables
gamma_df.to_csv(os.path.join(output_dir, "site_effects_raw_log_shift.csv"))
pct_shift_df.to_csv(os.path.join(output_dir, "site_effects_percentage.csv"))

# 5. Plot a Heatmap of the Percentage Shifts
plt.figure(figsize=(16, 8))
sns.set_theme(style="white", context="talk", font_scale=0.85)

# Use a diverging colormap (vlag: blue for negative/deflation, red for positive/inflation)
heatmap = sns.heatmap(
    pct_shift_df, 
    cmap="vlag", 
    center=0, 
    annot=True,          # Show the actual numbers in the boxes
    fmt=".1f",           # Format to 1 decimal place (e.g., "5.2")
    linewidths=.5, 
    cbar_kws={'label': '% Shift from Grand Mean'}
)

plt.title("Scanner Bias: Estimated Site Effects on Brain Metrics", pad=20, fontweight='bold')
plt.xlabel("Brain Metrics", labelpad=10)
plt.ylabel("Cohort / Site", labelpad=10)

# Rotate the x-axis labels so they don't overlap
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "scanner_bias_percentage_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Site effect quantification complete.")

# ==========================================
# 5. GLOBAL AESTHETIC SETTINGS FOR PLOTS
# ==========================================
sns.set_theme(style="ticks", context="talk", font_scale=0.9)
COLOR_RAW = "#3498db"     
COLOR_HARM = "#e67e22"    
PALETTE_BATCH = "Set2"    
CMAP_AGE = "mako"         

# ==========================================
# 6. GENERATE PLOTS
# ==========================================
for col in brain_cols:
    
    # --- Side-by-Side Scatter Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.scatterplot(
        ax=axes[0], x=data_df['gestational_age'], y=data_df[col],
        color=COLOR_RAW, alpha=0.7, edgecolor='white', linewidth=0.5, s=60
    )
    axes[0].set_title(f"Raw: {col}", pad=15, fontweight='bold')
    axes[0].set_xlabel("Gestational Age (weeks)", labelpad=10)
    axes[0].set_ylabel(col, labelpad=10)

    sns.scatterplot(
        ax=axes[1], x=data_df['gestational_age'], y=harmonized_df[f"{col}_harm"],
        color=COLOR_HARM, alpha=0.7, edgecolor='white', linewidth=0.5, s=60
    )
    axes[1].set_title(f"Harmonized: {col}", pad=15, fontweight='bold')
    axes[1].set_xlabel("Gestational Age (weeks)", labelpad=10)

    sns.despine() 
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}_vs_age_side_by_side_by_cohort.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Overlay Scatter Plot ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data_df['gestational_age'], y=data_df[col], 
        color=COLOR_RAW, label='Raw Data', alpha=0.6, edgecolor='white', linewidth=0.5, s=70
    )
    sns.scatterplot(
        x=data_df['gestational_age'], y=harmonized_df[f"{col}_harm"],
        color=COLOR_HARM, label='Harmonized Data', alpha=0.6, edgecolor='white', linewidth=0.5, s=70
    )
    plt.title(f"Raw vs Harmonized: {col}", pad=15, fontweight='bold')
    plt.xlabel("Gestational Age (weeks)", labelpad=10)
    plt.ylabel(col, labelpad=10)
    plt.legend(frameon=False, loc='best')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}_raw_and_harmonized_scatter_by_cohort.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Boxplots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.boxplot(
        ax=axes[0], x=data_df['batch'], y=data_df[col],
        color=COLOR_RAW, width=0.5, boxprops=dict(alpha=0.8), fliersize=3
    )
    axes[0].set_title(f"Raw by Cohort: {col}", pad=15, fontweight='bold')
    axes[0].set_xlabel("Cohort", labelpad=10)
    axes[0].set_ylabel(col, labelpad=10)
    axes[0].tick_params(axis='x', rotation=45)

    sns.boxplot(
        ax=axes[1], x=data_df['batch'], y=harmonized_df[f"{col}_harm"],
        color=COLOR_HARM, width=0.5, boxprops=dict(alpha=0.8), fliersize=3
    )
    axes[1].set_title(f"Harmonized by Cohort: {col}", pad=15, fontweight='bold')
    axes[1].set_xlabel("Cohort", labelpad=10)
    axes[1].tick_params(axis='x', rotation=45)

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{col}_boxplot_by_cohort.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 7. PCA PLOTS
# ==========================================
print("Generating PCA plots...")

# A. Create a scaler specifically for PCA
pca_scaler = StandardScaler()

# B. Standardize the data so large variables (like volume) don't dominate small variables (like percentages)
# We use .values on data_log to prevent column-name warnings from sklearn
original_scaled_for_pca = pca_scaler.fit_transform(data_log.values) 
combat_scaled_for_pca = pca_scaler.transform(harmonized_data_log) 

# C. Run PCA on the standardized data
pca = PCA(n_components=3)
pca_result_before = pca.fit_transform(original_scaled_for_pca)
pca_result_after = pca.transform(combat_scaled_for_pca)

df_pca_before = pd.DataFrame({
    "PC1": pca_result_before[:, 0], "PC2": pca_result_before[:, 1], "PC3": pca_result_before[:, 2],
    "batch": data_df["batch"], "gestational_age": data_df["gestational_age"]
})

df_pca_after = pd.DataFrame({
    "PC1": pca_result_after[:, 0], "PC2": pca_result_after[:, 1], "PC3": pca_result_after[:, 2],
    "batch": data_df["batch"], "gestational_age": data_df["gestational_age"]
})

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# --- Plot by Batch ---
sns.scatterplot(
    ax=axes[0, 0], x="PC1", y="PC2", hue="batch", data=df_pca_before, 
    palette=PALETTE_BATCH, alpha=0.8, edgecolor='white', s=60
)
axes[0, 0].set_title("Raw Data (By Cohort)", pad=15, fontweight='bold')
axes[0, 0].legend(frameon=False, title="Cohort")

sns.scatterplot(
    ax=axes[0, 1], x="PC1", y="PC2", hue="batch", data=df_pca_after, 
    palette=PALETTE_BATCH, alpha=0.8, edgecolor='white', s=60, legend=False
)
axes[0, 1].set_title("Harmonized Data (By Cohort)", pad=15, fontweight='bold')

# --- Plot by Age ---
scatter1 = axes[1, 0].scatter(
    df_pca_before["PC1"], df_pca_before["PC2"],
    c=df_pca_before["gestational_age"], cmap=CMAP_AGE, 
    alpha=0.8, edgecolors='white', linewidth=0.5, s=60
)
axes[1, 0].set_title("Raw Data (By Age)", pad=15, fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=axes[1, 0])
cbar1.set_label("Gestational Age (weeks)", rotation=270, labelpad=15)
cbar1.outline.set_visible(False) 

scatter2 = axes[1, 1].scatter(
    df_pca_after["PC1"], df_pca_after["PC2"], 
    c=df_pca_after["gestational_age"], cmap=CMAP_AGE,
    alpha=0.8, edgecolors='white', linewidth=0.5, s=60
)
axes[1, 1].set_title("Harmonized Data (By Age)", pad=15, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=axes[1, 1])
cbar2.set_label("Gestational Age (weeks)", rotation=270, labelpad=15)
cbar2.outline.set_visible(False)

sns.despine()
plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(plot_dir, "harmonization_effect_pca_by_cohort.png"), bbox_inches="tight", dpi=300)
plt.close()

print("All tasks finished successfully!")