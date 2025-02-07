import slam.io as sio
import slam.texture as stex
from slam.differential_geometry import laplacian_texture_smoothing
import matplotlib.pyplot as plt
import seaborn as sns

# Declare file paths
mesh_file = '/home/INT/dienye.h/Python Codes/rough_hemisphere/mesh_surfaces/sub-0001_ses-0001_reo-SVR-output-brain-mask-brain_bounti-white.left.surf.gii'
mean_curv_texture = '/home/INT/dienye.h/Python Codes/Curvature/mean curvature/filt_mean_curv_sub-0001_ses-0001_reo-SVR-output-brain-mask-brain_bounti-white.left.surf.gii'

# Load mesh and textures
mesh = sio.load_mesh(mesh_file)
mean_curv_tex = sio.load_texture(mean_curv_texture)
mean_curv = mean_curv_tex.darray.squeeze()

# Calculate smoothed versions with different iterations
mean_curv_smooth_5 = laplacian_texture_smoothing(mesh, mean_curv, nb_iter=5, dt=0.1)
mean_curv_smooth_10 = laplacian_texture_smoothing(mesh, mean_curv, nb_iter=10, dt=0.1)
mean_curv_smooth_20 = laplacian_texture_smoothing(mesh, mean_curv, nb_iter=20, dt=0.1)

# Define color scheme
colors = {
    'original': '#2E86C1',
    'iter5': '#E74C3C',
    'iter10': '#2ECC71',
    'iter20': '#F39C12'
}

# Function to create individual plots
def create_individual_plot(data, title, color, filename):
    plt.style.use('seaborn')
    plt.figure(figsize=(15, 8))
    
    sns.histplot(
        data=data,
        bins=200,
        color=color,
        alpha=0.7,
        stat='count'  # Changed to count for actual frequencies
    )
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Mean Curvature', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)  # Changed y-label to reflect count
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Create individual plots
create_individual_plot(
    mean_curv,
    'Frequency Distribution of Original Mean Curvature Values',
    colors['original'],
    '/home/INT/dienye.h/Python Codes/Curvature/analysis/mean_curvature_distribution_original.png'
)

create_individual_plot(
    mean_curv_smooth_5,
    'Frequency Distribution of Mean Curvature Values (5 iterations)',
    colors['iter5'],
    '/home/INT/dienye.h/Python Codes/Curvature/analysis/mean_curvature_distribution_iter5.png'
)

create_individual_plot(
    mean_curv_smooth_10,
    'Frequency Distribution of Mean Curvature Values (10 iterations)',
    colors['iter10'],
    '/home/INT/dienye.h/Python Codes/Curvature/analysis/mean_curvature_distribution_iter10.png'
)

create_individual_plot(
    mean_curv_smooth_20,
    'Frequency Distribution of Mean Curvature Values (20 iterations)',
    colors['iter20'],
    '/home/INT/dienye.h/Python Codes/Curvature/analysis/mean_curvature_distribution_iter20.png'
)

# Create combined plot
plt.style.use('seaborn')
plt.figure(figsize=(15, 8))

# Create histograms with different colors and labels
sns.histplot(
    data=mean_curv,
    bins=200,
    color=colors['original'],
    alpha=0.4,
    label='Original',
    stat='count'  # Changed to count for actual frequencies
)

sns.histplot(
    data=mean_curv_smooth_5,
    bins=200,
    color=colors['iter5'],
    alpha=0.4,
    label='5 iterations',
    stat='count'  # Changed to count for actual frequencies
)

sns.histplot(
    data=mean_curv_smooth_10,
    bins=200,
    color=colors['iter10'],
    alpha=0.4,
    label='10 iterations',
    stat='count'  # Changed to count for actual frequencies
)

sns.histplot(
    data=mean_curv_smooth_20,
    bins=200,
    color=colors['iter20'],
    alpha=0.4,
    label='20 iterations',
    stat='count'  # Changed to count for actual frequencies
)

# Add title and labels
plt.title('Frequency Distribution of Mean Curvature Values: Original vs Smoothened', fontsize=14, pad=15)
plt.xlabel('Mean Curvature', fontsize=12)
plt.ylabel('Frequency', fontsize=12)  # Changed y-label to reflect count

# Add legend
plt.legend(title='Smoothing Iterations', fontsize=10, title_fontsize=11)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save combined plot
plt.savefig('/home/INT/dienye.h/Python Codes/Curvature/analysis/mean_curvature_distribution_combined.png', dpi=300)

# Close the plot
plt.close()