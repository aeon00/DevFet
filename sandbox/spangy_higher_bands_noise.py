# import numpy as np
# import pandas as pd
# import os
# from glob import glob

# # Directory containing frecomposed textures
# frecomposed_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/marsfet_full_info/frecomposed/full/'
# output_csv = '/envau/work/meca/users/dienye.h/meso_envau_sync/marsfet_full_info/info/power_above_b6_results.csv'

# # Find all .npy files in directory
# texture_files = sorted(glob(os.path.join(frecomposed_dir, '*.npy')))

# print(f"Found {len(texture_files)} texture files")

# # Store results
# results = []

# # Process each texture file
# for texture_path in texture_files:
#     # Extract subject/filename identifier
#     filename = os.path.basename(texture_path)
#     subject_id = os.path.splitext(filename)[0]  # Remove .npy extension
    
#     print(f"\nProcessing: {subject_id}")
    
#     try:
#         # Load the frecomposed array (allow_pickle=True for object arrays)
#         frecomposed = np.load(texture_path, allow_pickle=True)
        
#         # Convert object array to numeric array if needed
#         if frecomposed.dtype == 'object':
#             print(f"  Converting from object array to numeric array")
#             frecomposed = np.array(frecomposed.tolist(), dtype=float)
        
#         # Check shape
#         print(f"  Shape: {frecomposed.shape}")
        
#         # Get number of bands
#         nlevels = frecomposed.shape[1] + 1
        
#         # Compute power for all bands
#         all_band_powers = np.zeros(frecomposed.shape[1])
#         for i in range(frecomposed.shape[1]):
#             all_band_powers[i] = np.sum(frecomposed[:, i]**2)
        
#         # Compute AFP (excluding B0)
#         afp_all = np.sum(all_band_powers[1:])
        
#         # Sum of power in all bands above B6
#         power_above_b6 = np.sum(all_band_powers[7:])
#         relative_power_above_b6 = power_above_b6 / afp_all if afp_all > 0 else 0
        
#         # Store results
#         result_dict = {
#             'subject_id': subject_id,
#             'total_power_above_b6': power_above_b6,
#             'relative_power_above_b6': relative_power_above_b6,
#             'total_afp': afp_all,
#             'n_bands': frecomposed.shape[1]
#         }
        
#         # Optionally store individual band powers above B6
#         for i in range(7, len(all_band_powers)):
#             result_dict[f'B{i}_power'] = all_band_powers[i]
#             result_dict[f'B{i}_relative'] = all_band_powers[i] / afp_all if afp_all > 0 else 0
        
#         results.append(result_dict)
        
#         print(f"  Total power B7+: {power_above_b6:.6f}")
#         print(f"  Relative power B7+: {relative_power_above_b6:.6f}")
        
#     except Exception as e:
#         print(f"  ERROR processing {subject_id}: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         continue

# # Create DataFrame and save to CSV
# df_results = pd.DataFrame(results)
# df_results.to_csv(output_csv, index=False)

# print(f"\n{'='*60}")
# print(f"Processing complete!")
# print(f"Total subjects processed: {len(results)}")
# print(f"Results saved to: {output_csv}")
# print(f"{'='*60}")

# # Display summary statistics
# if len(results) > 0:
#     print("\n** Summary Statistics **")
#     print(df_results[['total_power_above_b6', 'relative_power_above_b6', 'total_afp']].describe())

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

def process_single_file(filename, surface_path, df):
    """
    Process a single surface file and compute various metrics at different smoothing levels.
    """
    try:
        start_time = time.time()
        print("Starting processing of {}".format(filename))
        
        hemisphere = 'left' if filename.endswith('left.surf.gii') else 'right'
        participant_session = filename.split('_')[0] + '_' + filename.split('_')[1] + f'_{hemisphere}'
        base_participant_session = filename.split('_')[0] + '_' + filename.split('_')[1]
        
        # Get corresponding gestational age
        try:
            gestational_age = df[df['participant_session'] == base_participant_session]['fetus_gestational_age_at_scan'].values[0]
        except:
            print(f"Warning: No matching gestational age found for {base_participant_session}")
            return None
        
        mesh_file = os.path.join(surface_path, filename)
        if not os.path.exists(mesh_file):
            print("Error: Mesh file not found: {}".format(mesh_file))
            return None
        
        # Load the base mesh once
        base_mesh = sio.load_mesh(mesh_file)
        
        # Define smoothing iterations to test
        smoothing_levels = [0, 5, 10, 20]
        results = []
        
        for smooth_iter in smoothing_levels:
            print(f"  Processing with {smooth_iter} smoothing iterations...")
            iter_start_time = time.time()
            
            # Apply smoothing if needed
            if smooth_iter == 0:
                mesh = base_mesh.copy()
            else:
                mesh = laplacian_mesh_smoothing(base_mesh, nb_iter=smooth_iter, dt=0.1)
            
            # Apply principal inertia transform
            mesh.apply_transform(mesh.principal_inertia_transform)
            
            N = 5000
            
            # Compute eigenpairs and mass matrix
            print(f"    Computing eigen vectors and eigen values")
            eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)
            
            # CURVATURE
            print(f"    Computing mean curvature")
            PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
                scurv.curvatures_and_derivatives(mesh)
            mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
            
            tex_mean_curv = stex.TextureND(mean_curv)
            tex_mean_curv.z_score_filtering(z_thresh=3)
            filt_mean_curv = tex_mean_curv.darray.squeeze()
            
            # WHOLE BRAIN MEAN-CURVATURE SPECTRUM
            grouped_spectrum, group_indices, coefficients, nlevels \
                = spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
            
            # Compute total AFP (excluding B0)
            afp_all = np.sum(grouped_spectrum[1:])
            
            # Compute power above B6
            power_above_b6 = np.sum(grouped_spectrum[7:])  # B7 onwards
            relative_power_above_b6 = power_above_b6 / afp_all if afp_all > 0 else 0
            
            # Print for verification
            print(f"    Total bands: {len(grouped_spectrum)}")
            print(f"    Power above B6: {power_above_b6:.6f}")
            print(f"    Relative power above B6: {relative_power_above_b6:.6f}")
            
            # Store results
            result = {
                'subject_id': participant_session,
                'gestational_age': gestational_age,
                'hemisphere': hemisphere,
                'smoothing_iterations': smooth_iter,
                'total_power_above_b6': power_above_b6,
                'relative_power_above_b6': relative_power_above_b6,
                'total_afp': afp_all,
                'n_bands': len(grouped_spectrum),
                'processing_time': time.time() - iter_start_time
            }
            
            # Optionally add individual band powers
            for i in range(len(grouped_spectrum)):
                result[f'B{i}_power'] = grouped_spectrum[i]
            
            results.append(result)
        
        print(f"Total processing time for {filename}: {time.time() - start_time:.2f}s")
        return results
        
    except Exception as e:
        print("Error processing {}: {}".format(filename, str(e)))
        import traceback
        traceback.print_exc()
        return None
    
def main():
    try:
        # Paths
        surface_path = "/envau/work/meca/users/dienye.h/rough/spangy/noise_test/marsfet/mesh"
        mesh_info_path = "/envau/work/meca/users/dienye.h/meso_envau_sync/marsFet_HASTE_lastest_volumes_BOUNTI.csv"
        
        
        print("Reading data from {}".format(mesh_info_path))
        # Read dataframe
        df = pd.read_csv(mesh_info_path, sep='\t')
        df['participant_session'] = df['participant_id'] + '_' + df['session_id']
        
        print("Scanning directory: {}".format(surface_path))
        # Get list of files
        all_files = [f for f in os.listdir(surface_path) if f.endswith('left.surf.gii') or f.endswith('right.surf.gii')]
        print("Found {} files to process".format(len(all_files)))
        
        # Process directly if not in SLURM environment
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            print("Running in local mode - processing all files")
            results = []
            for filename in all_files:
                result = process_single_file(filename, surface_path, df)
                if result:
                    results.append(result)
                    
            if results:
                results_df = pd.DataFrame(results)
                output_file = os.path.join('/envau/work/meca/users/dienye.h/rough/spangy/noise_test/marsfet/', 'all_results.csv')
                results_df.to_csv(output_file, index=False)
                print(f"All results saved to {output_file}")
            else:
                print("Warning: No results generated")
            return
                
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
            if result:
                results.append(result)
        
        # Save results for this chunk
        if results:
            results_df = pd.DataFrame(results)
            chunk_file = os.path.join('/envau/work/meca/users/dienye.h/rough/spangy/noise_test/marsfet/', 'chunk_{}_results.csv'.format(task_id))
            results_df.to_csv(chunk_file, index=False)
            print("Results for chunk {} saved to {}".format(task_id, chunk_file))
        else:
            print("Warning: No results generated for chunk {}".format(task_id))
            
    except Exception as e:
        print("Critical error in main: {}".format(str(e)))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()