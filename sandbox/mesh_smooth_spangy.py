import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
import os
import pandas as pd
import time
import slam.spangy as spgy
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import sys
from slam.differential_geometry import laplacian_mesh_smoothing

# Function to get hull area

def get_hull_area(mesh):
    convex_hull = trimesh.convex.convex_hull(mesh)
    return float(convex_hull.area)  # Convert to float to ensure it's a numeric value

def get_gyrification_index(mesh):
    """
    Gyrification index as surface ratio between a mesh and its convex hull
    
    Parameters
    ----------
    mesh : Trimesh
        Triangular watertight mesh
    
    Returns
    -------
    gyrification_index : Float
        surface ratio between a mesh and its convex hull
    hull_area : Float
        area of the convex hull
    """
    hull_area = get_hull_area(mesh)
    gyrification_index = float(mesh.area) / hull_area  # Convert both to float
    return gyrification_index, hull_area


def process_single_file(filename, surface_path, df):
    """
    Process a single surface file and compute various metrics.
    """
    try:
        start_time = time.time()
        print("Starting processing of {}".format(filename))
    

        mesh_file = os.path.join(surface_path, filename)
        if not os.path.exists(mesh_file):
            print("Error: Mesh file not found: {}".format(mesh_file))
            return None
            
        mesh = sio.load_mesh(mesh_file)
        mesh = laplacian_mesh_smoothing(mesh, nb_iter=5, dt=0.1)
        filename = str(filename) + '_smooth_5'
        new_mesh_path = os.path.join('/envau/work/meca/users/dienye.h/smoothened_mesh_data/mesh/', '{}.gii'.format(filename))
        sio.write_mesh(mesh, new_mesh_path)
        mesh.apply_transform(mesh.principal_inertia_transform)
        N = 5000
        
        # Compute eigenpairs and mass matrix
        print("compute the eigen vectors and eigen values")
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

        # CURVATURE
        print("compute the mean curvature")
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)
        tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
        principal_tex_path = os.path.join('/envau/work/meca/users/dienye.h/smoothened_mesh_data/principal_curvature/', 'principal_curv_{}.gii'.format(filename))
        sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        mean_tex_path = os.path.join('/envau/work/meca/users/dienye.h/smoothened_mesh_data/mean_curvature/', 'filt_mean_curv_{}.gii'.format(filename))
        sio.write_texture(tex_mean_curv, mean_tex_path)
        filt_mean_curv = tex_mean_curv.darray.squeeze()
        total_mean_curv = sum(filt_mean_curv)
        # gyral_mask = np.where(filt_mean_curv > 0, 0, filt_mean_curv) # To mask gyri and only focus on sulci

        # WHOLE BRAIN MEAN-CURVATURE SPECTRUM
        grouped_spectrum, group_indices, coefficients, nlevels \
            = spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
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
        fig.savefig(f'/envau/work/meca/users/dienye.h/smoothened_mesh_data/spangy_plot_results/{filename}.png', bbox_inches='tight', dpi=300)
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

        # b. Band number of parcels
        print('** b. Band number of parcels **')
        print('B4 = %f, B5 = %f, B6 = %f' % (0, 0, 0))

        # c. Band power
        print('** c. Band power **')
        print('B4 = %f, B5 = %f, B6 = %f' %
            (grouped_spectrum[4], grouped_spectrum[5],
            grouped_spectrum[6]))

        # d. Band relative power
        print('** d. Band relative power **')
        print('B4 = %0.5f, B5 = %0.5f , B6 = %0.5f' %
            (grouped_spectrum[4] / afp, grouped_spectrum[5] / afp,
            grouped_spectrum[6] / afp))

        # LOCAL SPECTRAL BANDS
        loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, filt_mean_curv,
                                                            levels, group_indices,
                                                            eigVects)

        tex_path = f"/envau/work/meca/users/dienye.h/smoothened_mesh_data/spangy_textures/dominant_band/spangy_dom_band_{filename}.gii"
        tmp_tex = stex.TextureND(loc_dom_band)
        # tmp_tex.z_score_filtering(z_thresh=3)
        sio.write_texture(tmp_tex, tex_path)
        
        # Get hull area and gyrification index
        gyrification_index, hull_area = get_gyrification_index(mesh)

        # Calculate and print execution time for this iteration
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time for {filename}: {execution_time:.2f} seconds")
        print("----------------------------------------\n")
        
        return {

        }
        
    except Exception as e:
        print("Error processing {}: {}".format(filename, str(e)))
        return None

def main():
    try:
        # Paths
        surface_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_surfaces/haste/"
        mesh_info_path = "/envau/work/meca/users/dienye.h/bounti_info/marsFet_HASTE_lastest_volumes_BOUNTI.csv"
        
        print("Reading data from {}".format(mesh_info_path))
        # Read dataframe
        df = pd.read_csv(mesh_info_path)
        df['participant_session'] = df['participant_id'] + '_' + df['session_id']
        
        print("Scanning directory: {}".format(surface_path))
        # Get list of files
        all_files = [f for f in os.listdir(surface_path) if f.endswith('left.surf.gii') or f.endswith('right.surf.gii')]
        print("Found {} files to process".format(len(all_files)))
        
        # Calculate chunk for this array task
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        chunk_size = len(all_files) // n_tasks + (1 if len(all_files) % n_tasks > 0 else 0)
        start_idx = task_id * chunk_size
        end_idx = min((task_id + 1) * chunk_size, len(all_files))
        
        print("Processing chunk {}/{} (files {} to {})".format(task_id + 1, n_tasks, start_idx, end_idx))
        
        # Process files in this chunk
        results = []
        for filename in all_files[start_idx:end_idx]:
            result = process_single_file(filename, surface_path, df)
            
    except Exception as e:
        print("Critical error in main: {}".format(str(e)))
        sys.exit(1)

if __name__ == "__main__":
    main()

