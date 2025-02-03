import slam.io as sio
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

directory = '/envau/work/meca/data/Fetus/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_surfaces/haste/'  # Add your directory path here
vertices_counts = [] 
surface_area_values = []
volume_values = []

# Define measurement units
mL_in_MM3 = 1000
CM2_in_MM2 = 100

# Collect vertex counts from all meshes
for filename in os.listdir(directory):
    if filename.endswith('surf.gii'): 
        mesh_file = os.path.join(directory, filename)
        mesh = sio.load_mesh(mesh_file)
        num_vertices = len(mesh.vertices)
        mL_in_MM3 = 1000
        CM2_in_MM2 = 100
        volume = mesh.volume
        volume = np.floor(volume / mL_in_MM3),
        surface_area = mesh.area,
        surface_area = np.floor(surface_area / CM2_in_MM2)
        vertices_counts.append(num_vertices)
        surface_area_values.append(surface_area)
        volume_values.append(volume)

# Create DataFrame
vertices_df = pd.DataFrame(vertices_counts, columns=['Number of Vertices'])
vertices_df.to_csv('/envau/work/meca/users/dienye.h/bounti_analysis/vertices_counts.csv', index=False)

surface_area_df = pd.DataFrame(surface_area_values, columns=['Surface Area Values'])
surface_area_df.to_csv('/envau/work/meca/users/dienye.h/bounti_analysis/surface_area_values.csv', index=False)

volume_df = pd.DataFrame(volume_values, columns=['Volume Values'])
volume_df.to_csv('/envau/work/meca/users/dienye.h/bounti_analysis/volume_values.csv', index=False)


# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))

# Create histogram of vertices
sns.histplot(
    data=vertices_df,
    x='Number of Vertices',  # Match the DataFrame column name
    bins=30,
    color='#2E86C1',        # Slightly darker blue for better visibility
    alpha=0.8,
    kde=True,
    line_kws={'color': '#E74C3C', 'linewidth': 2}
)

# Add title and labels
plt.title('Distribution of Vertex Counts Across Meshes', fontsize=14, pad=15)
plt.xlabel('Number of Vertices', fontsize=12)
plt.ylabel('Count of Meshes', fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Format x-axis with thousand separators
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adjust layout
plt.tight_layout()

# Save Plot
plt.savefig('/envau/work/meca/users/dienye.h/bounti_analysis/distribution_of_vertices.png') 

# Close the plot
plt.close()


# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))

# Create histogram of surface area values
sns.histplot(
    data=surface_area_df,
    x='Surface Area Values',  # Match the DataFrame column name
    bins=30,
    color='#2E86C1',        # Slightly darker blue for better visibility
    alpha=0.8,
    kde=True,
    line_kws={'color': '#E74C3C', 'linewidth': 2}
)

# Add title and labels
plt.title('Distribution of Surface Area Values Across Meshes', fontsize=14, pad=15)
plt.xlabel('Surface Area Values', fontsize=12)
plt.ylabel('Count of Meshes', fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Format x-axis with thousand separators
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adjust layout
plt.tight_layout()

# Save Plot
plt.savefig('/envau/work/meca/users/dienye.h/bounti_analysis/distribution_of_surface_area.png') 

# Close the plot
plt.close()


# Set the style
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))

# Create histogram of volume values
sns.histplot(
    data=volume_df,
    x='Volume Values',  # Match the DataFrame column name
    bins=30,
    color='#2E86C1',        # Slightly darker blue for better visibility
    alpha=0.8,
    kde=True,
    line_kws={'color': '#E74C3C', 'linewidth': 2}
)

# Add title and labels
plt.title('Distribution of Volume Values Across Meshes', fontsize=14, pad=15)
plt.xlabel('Volume Values', fontsize=12)
plt.ylabel('Count of Meshes', fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Format x-axis with thousand separators
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adjust layout
plt.tight_layout()

# Save Plot
plt.savefig('/envau/work/meca/users/dienye.h/bounti_analysis/distribution_of_volume.png') 

# Close the plot
plt.close()