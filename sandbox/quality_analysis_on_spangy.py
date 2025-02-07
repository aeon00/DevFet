import slam.io as sio
import slam.texture as stex
from slam.differential_geometry import laplacian_texture_smoothing
import matplotlib.pyplot as plt
import seaborn as sns
import slam.spangy as spgy
import numpy as np
import pandas as pd
import re
import os

def spangy_analysis(mesh_path, mesh, mean_curv, N, nb_smooth_iter):

    mesh = mesh
    N = N
    file = re.search(r'sub-\d+_ses-\d+', os.path.basename(mesh_path)).group(0)
    filename = file + "_" + str(N) + "_" + str(nb_smooth_iter)

    # Compute eigenpairs and mass matrix
    print("compute the eigen vectors and eigen values")
    eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

    # WHOLE BRAIN MEAN-CURVATURE SPECTRUM
    grouped_spectrum, group_indices, coefficients, nlevels \
        = spgy.spectrum(mean_curv, lap_b, eigVects, eigVal)
    levels = len(group_indices)

    # Plot coefficients and bands for all mean curvature signal
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    frequency = np.sqrt(eigVal/2*np.pi) # from equations in Appendix A.1
    ax1.scatter(frequency,
    coefficients, marker='o', s=20, linewidths=0.5)
    ax1.set_xlabel('Frequency (m⁻¹)')
    ax1.set_ylabel('Coefficients')

    ax2.scatter(frequency[1:],
    coefficients[1:], marker='o', s=20, linewidths=0.5) # remove B0 coefficients
    ax2.set_xlabel('Frequency (m⁻¹)')
    ax2.set_ylabel('Coefficients')

    ax3.bar(np.arange(0, levels), grouped_spectrum)
    ax3.set_xlabel('Spangy Frequency Bands')
    ax3.set_ylabel('Power Spectrum')
    plt.tight_layout()  # Adjust the spacing between subplots
    fig.savefig(f'/envau/work/meca/users/dienye.h/N_analysis/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    # a. Whole brain parameters
    mL_in_MM3 = 1000
    CM2_in_MM2 = 100
    volume = mesh.volume
    surface_area = mesh.area
    afp = np.sum(grouped_spectrum[1:])
    print('** a. Whole brain parameters **')
    print('Volume = %d mL, Area = %d cm², Analyze Folding Power = %f,' %
        (np.floor(volume / mL_in_MM3), np.floor(surface_area / CM2_in_MM2), afp))

    # Check if B6 exists in grouped_spectrum
    b6_value = grouped_spectrum[6] if len(grouped_spectrum) > 6 else 0

    # b. Band number of parcels
    print('** b. Band number of parcels **')
    print('B4 = %f, B5 = %f, B6 = %f' % (0, 0, 0))

    # c. Band power
    print('** c. Band power **')
    print('B4 = %f, B5 = %f, B6 = %f' %
        (grouped_spectrum[4], grouped_spectrum[5], b6_value))

    # d. Band relative power
    print('** d. Band relative power **')
    print('B4 = %0.5f, B5 = %0.5f , B6 = %0.5f' %
        (grouped_spectrum[4] / afp, grouped_spectrum[5] / afp,
        b6_value / afp if b6_value != 0 else 0))

    # LOCAL SPECTRAL BANDS
    loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, mean_curv,
                                                        levels, group_indices,
                                                        eigVects)

    tex_path = f"/envau/work/meca/users/dienye.h/N_analysis/spangy_dom_band_{filename}.gii"
    tmp_tex = stex.TextureND(loc_dom_band)
    # tmp_tex.z_score_filtering(z_thresh=3)
    sio.write_texture(tmp_tex, tex_path)

    return {
        'N' : N,
        'Number of smoothing iterations': nb_smooth_iter,
        'band_power_B0': grouped_spectrum[0],
        'band_power_B1': grouped_spectrum[1],
        'band_power_B2': grouped_spectrum[2],
        'band_power_B3': grouped_spectrum[3],
        'band_power_B4': grouped_spectrum[4],
        'band_power_B5': grouped_spectrum[5],
        'band_power_B6': b6_value,
        'volume_ml': np.floor(volume / mL_in_MM3),
        'surface_area_cm2': np.floor(surface_area / CM2_in_MM2),
        'analyze_folding_power': afp
    }

# Rest of the code remains unchanged
mesh_file = '/envau/work/meca/users/dienye.h/rough_hemisphere/mesh_surfaces/sub-0001_ses-0001_reo-SVR-output-brain-mask-brain_bounti-white.right.surf.gii'
mean_curv_texture = '/envau/work/meca/users/dienye.h/rough_hemisphere/mean_curv/filt_mean_curv_sub-0001_ses-0001_reo-SVR-output-brain-mask-brain_bounti-white.right.surf.gii'
subject= re.search(r'sub-\d+_ses-\d+', os.path.basename(mesh_file)).group(0)
# Load mesh and textures
mesh = sio.load_mesh(mesh_file)
mean_curv_tex = sio.load_texture(mean_curv_texture)
mean_curv = mean_curv_tex.darray.squeeze()

# Calculate smoothed versions with different iterations
mean_curv_smooth_5 = laplacian_texture_smoothing(mesh, mean_curv, nb_iter=5, dt=0.1)
mean_curv_smooth_10 = laplacian_texture_smoothing(mesh, mean_curv, nb_iter=10, dt=0.1)
mean_curv_smooth_20 = laplacian_texture_smoothing(mesh, mean_curv, nb_iter=20, dt=0.1)

mean_curv_lst = [mean_curv_smooth_5, mean_curv_smooth_10, mean_curv_smooth_20]

results = []

for i in range(1000, 6001, 500):
    N = i 
    nb_smooth_iter = 0
    for smoothed_curv in mean_curv_lst:
        nb_smooth_iter +=5 
        data = spangy_analysis(mesh_file, mesh, smoothed_curv, N, nb_smooth_iter)
        results.append(data)

results_df = pd.DataFrame(results)
results_df.to_csv(f'/envau/work/meca/users/dienye.h/N_analysis/QA_{subject}_results.csv', index=False)
