##Change if want to do by site (scanner)




import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler




from neuroHarmonize import harmonizationLearn

import os







input_file = "/home/INT/dienye.h/python_files/combined_dataset/combined_qc_dataset_with_site.csv"

output_dir = "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization"

plot_dir = "//home/INT/dienye.h/gamlss_normative_paper-main/harmonization/plots/by_cohort/"

os.makedirs(output_dir, exist_ok=True)




data_df = pd.read_csv(input_file)




#data_df["ICV"] = ( data_df["CSF.Volume"] + data_df["cGrey.Matter.Volume"] + data_df["White.Matter.Volume"] + data_df["Lateral.Ventricles.Volume"] + data_df["Cerebellum.Volume"] + data_df["Basal.Ganglia.Volume"] + data_df["Thalamus.Volume"]+ data_df["Brainstem.Volume"] )




data_df['batch'] = data_df['SITE']  #convert cohort variable to BATCH variable for analysis. Can change to "scanner_site".




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




data = data_df[brain_cols]

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)




#Apply neuroHarmonize

covars_harm = covars.copy()

covars_harm['SITE'] = covars_harm['batch'].astype(str)

covars_harm = covars_harm.drop(columns=['batch'])




data_array = data_scaled


estimates, harmonized_data, s_data = harmonizationLearn(

    data_array, covars=covars_harm, smooth_terms=["age"], return_s_data=True,
ref_batch="0", eb=True

)




harmonized_df = data_df.copy()

harmonized_columns = {col: f"{col}_harm" for col
in brain_cols}

harmonized_data_df = pd.DataFrame(harmonized_data, columns=[harmonized_columns[col] for col in brain_cols])




#Rescale harmonized data to the same scale as pre-harmonized data

harmonized_data_df = scaler.inverse_transform(harmonized_data)

rescaled_harmonized_df = pd.DataFrame(harmonized_data_df, columns=brain_cols)




harmonized_df = pd.concat([harmonized_df, rescaled_harmonized_df.add_suffix("_harm")], axis=1)

harmonized_df.to_csv(os.path.join(output_dir, "harmonized_data_dhcp_marsfet.csv"), index=False)







#Plots

for col in brain_cols:

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
sharey=True)

    #Scatter

    sns.scatterplot(ax=axes[0], x=data_df['gestational_age'], y=data_df[col],
color='blue')

    axes[0].set_title(f"Raw: {col} vs Gestational Age")

    axes[0].set_xlabel("Gestational Age")

    axes[0].set_ylabel(col)




    sns.scatterplot(ax=axes[1], x=data_df['gestational_age'], y=harmonized_df[f"{col}_harm"],
color='orange')

    axes[1].set_title(f"Harmonized: {col} vs Gestational Age")

    axes[1].set_xlabel("Gestational Age")




    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"{col}_vs_age_side_by_side_by_cohort.png"), dpi=300)

    plt.close()

# Scatter plot for raw and harmonized data

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=data_df['gestational_age'], y=data_df[col], color='blue',
label='Raw Data')

    sns.scatterplot(x=data_df['gestational_age'], y=harmonized_df[f"{col}_harm"],
color='orange', label='Harmonized Data')

    plt.title(f"Raw and Harmonized: {col} vs Gestational Age")

    plt.xlabel("Age")

    plt.ylabel(col)

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"{col}_raw_and_harmonized_scatter_by_cohort.png"), dpi=300)

    plt.close()

    

    #Boxplot

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
sharey=True)




    sns.boxplot(ax=axes[0], x=data_df['batch'], y=data_df[col],
color='blue')

    axes[0].set_title(f"Raw: {col} by cohort")

    axes[0].set_xlabel("cohort")

    axes[0].set_ylabel(col)

    axes[0].tick_params(axis='x', rotation=45)




    sns.boxplot(ax=axes[1], x=data_df['batch'], y=harmonized_df[f"{col}_harm"],
color='orange')

    axes[1].set_title(f"Harmonized: {col} by cohort")

    axes[1].set_xlabel("cohort")

    axes[1].tick_params(axis='x', rotation=45)




    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, f"{col}_boxplot_by_cohort.png"), dpi=300)

    plt.close()




#PCA plots

pca = PCA(n_components=3)

original_scaled = scaler.fit_transform(data)

combat_scaled = scaler.transform(harmonized_data)




pca_result_before = pca.fit_transform(original_scaled)

pca_result_after = pca.transform(combat_scaled)




#DataFrame with PCA results

df_pca_before = pd.DataFrame(

    {

        "PC1": pca_result_before[:, 0],

        "PC2": pca_result_before[:, 1],

        "PC3": pca_result_before[:, 2],

        "batch": data_df["batch"],

        "gestational_age": data_df["gestational_age"],

    }

)




df_pca_after = pd.DataFrame(

    {

        "PC1": pca_result_after[:, 0],

        "PC2": pca_result_after[:, 1],

        "PC3": pca_result_after[:, 2],

        "batch": data_df["batch"],

        "gestational_age": data_df["gestational_age"],

    }

)




fig, axes = plt.subplots(2, 2, figsize=(15, 15))




#Plot by batch

sns.scatterplot(

    ax=axes[0, 0], x="PC1", y="PC2",
hue="batch", data=df_pca_before, palette="viridis"

)

axes[0, 0].set_title("Raw Data (cohort)")




sns.scatterplot(

    ax=axes[0, 1], x="PC1", y="PC2",
hue="batch", data=df_pca_after, palette="viridis",
legend=False

)

axes[0, 1].set_title("After Harmonization (cohort)")




#Plot by age

scatter1 = axes[1, 0].scatter(

    df_pca_before["PC1"],

    df_pca_before["PC2"],

    c=df_pca_before["gestational_age"],

    cmap="viridis",

)

axes[1, 0].set_title("Raw Data (Gestational Age)")

plt.colorbar(scatter1, ax=axes[1, 0], label="Gestational Age (weeks)")




scatter2 = axes[1, 1].scatter(

    df_pca_after["PC1"], df_pca_after["PC2"], c=df_pca_after["gestational_age"], cmap="viridis"

)

axes[1, 1].set_title("After Harmonization (Gestational Age)")

plt.colorbar(scatter2, ax=axes[1, 1], label="Gestational Age (weeks)")




plt.tight_layout()

plt.savefig(

    os.path.join(plot_dir, "harmonization_effect_pca_by_cohort.png"), bbox_inches="tight", dpi=300

)

plt.close()