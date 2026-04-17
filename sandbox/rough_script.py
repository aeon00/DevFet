# # import slam.io as sio
# # import slam.texture as stex
# # import slam.curvature as scurv
# # import os
# # from slam.differential_geometry import laplacian_mesh_smoothing

# # def ensure_dir_exists(directory):
# #     """Create directory if it doesn't exist"""
# #     if not os.path.exists(directory):
# #         os.makedirs(directory)

# # mesh_dir = '/envau/work/meca/users/dienye.h/qc_identified_meshes/mesh_no_curvature'
# # mean_tex_dir = '/envau/work/meca/users/dienye.h/qc_identified_meshes/mesh_no_curvature'
# # principal_tex_dir = '/envau/work/meca/users/dienye.h/qc_identified_meshes/mesh_no_curvature'

# # # Ensure output directories exist
# # ensure_dir_exists(mean_tex_dir)
# # ensure_dir_exists(principal_tex_dir)

# # for mesh_file in os.listdir(mesh_dir):
# #     if not mesh_file.endswith(('.gii', '.mesh', '.ply')):  # Add appropriate mesh extensions
# #         continue
    
# #     # Extract filename without extension for texture naming
# #     filename = os.path.splitext(mesh_file)[0]
    
# #     # Define the path where the mean curvature texture would be saved
# #     mean_tex_path = os.path.join(mean_tex_dir, 'filt_mean_curv_{}.gii'.format(filename))
    
# #     # Check if mean curvature texture already exists
# #     if os.path.exists(mean_tex_path):
# #         print(f"Mean curvature texture already exists for {mesh_file}, skipping...")
# #         continue
    
# #     print(f"Processing mesh: {mesh_file}")
    
# #     # Load and process the mesh
# #     mesh_path = os.path.join(mesh_dir, mesh_file)
# #     mesh = sio.load_mesh(mesh_path)
    
# #     # CURVATURE
# #     print("compute the mean curvature")
# #     PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
# #         scurv.curvatures_and_derivatives(mesh)
# #     # tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
    
# #     # # Save principal curvature texture
# #     # principal_tex_path = os.path.join(principal_tex_dir, 'principal_curv_{}.gii'.format(filename))
# #     # sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
    
# #     # Compute and save mean curvature texture
# #     mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
# #     print('it works')
# #     # tex_mean_curv = stex.TextureND(mean_curv)
# #     # tex_mean_curv.z_score_filtering(z_thresh=3)
    
# #     # sio.write_texture(tex_mean_curv, mean_tex_path)
# #     print(f"Completed processing: {mesh_file}")

# # print("All meshes processed!")

# import slam.io as sio
# import slam.texture as stex
# import slam.curvature as scurv
# import os
# import pandas as pd
# import time
# import slam.spangy as spgy
# from slam.differential_geometry import laplacian_mesh_smoothing
# import numpy as np
# import matplotlib.pyplot as plt
# import trimesh
# import sys

# # Function to get hull area
# def get_hull_area(mesh):
#     convex_hull = trimesh.convex.convex_hull(mesh)
#     return float(convex_hull.area)  # Convert to float to ensure it's a numeric value

# def get_gyrification_index(mesh):
#     """
#     Gyrification index as surface ratio between a mesh and its convex hull
    
#     Parameters
#     ----------
#     mesh : Trimesh
#         Triangular watertight mesh
    
#     Returns
#     -------
#     gyrification_index : Float
#         surface ratio between a mesh and its convex hull
#     hull_area : Float
#         area of the convex hull
#     """
#     hull_area = get_hull_area(mesh)
#     gyrification_index = float(mesh.area) / hull_area  # Convert both to float
#     return gyrification_index, hull_area

# def calculate_band_wavelength(eigVal, group_indices):
#     """
#     Calculate the wavelength for each band based on eigenvalues
    
#     Parameters
#     ----------
#     eigVal : ndarray
#         Eigenvalues from the spectrum analysis
#     group_indices : list of arrays
#         Indices of eigenvalues in each band
    
#     Returns
#     -------
#     band_wavelengths : list
#         Average wavelength for each band in mm
#     """
#     band_wavelengths = []
    
#     for indices in group_indices:
#         if len(indices) == 0:  # Check if array is empty
#             band_wavelengths.append(0)
#             continue
            
#         # Calculate frequency from eigenvalues using equation in Appendix A.1
#         band_frequencies = np.sqrt(eigVal[indices] / (2 * np.pi))
        
#         # Wavelength = 1/frequency (in mm)
#         if np.mean(band_frequencies) > 0:
#             avg_wavelength = 1 / np.mean(band_frequencies)
#         else:
#             avg_wavelength = 0
            
#         band_wavelengths.append(avg_wavelength)
    
#     return band_wavelengths

# def calculate_parcels_per_band(loc_dom_band, levels):
#     """
#     Calculate the number of parcels (connected components) for each band
    
#     Parameters
#     ----------
#     loc_dom_band : ndarray
#         Local dominant band map
#     levels : int
#         Number of frequency bands
    
#     Returns
#     -------
#     parcels_per_band : list
#         Number of parcels for each band
#     """
#     from scipy import ndimage
    
#     parcels_per_band = []
    
#     for i in range(levels):
#         # Create binary mask for this band
#         band_mask = (loc_dom_band == i).astype(int)
        
#         # Check if the band exists at all
#         if np.sum(band_mask) == 0:
#             parcels_per_band.append(0)
#             continue
            
#         # Label connected components
#         labeled_array, num_parcels = ndimage.label(band_mask)
        
#         parcels_per_band.append(num_parcels)
    
#     return parcels_per_band

# def calculate_band_coverage(mesh, loc_dom_band, band_idx):
#     """
#     Calculate the coverage metrics for a specific frequency band using local dominance map.
    
#     Parameters:
#     mesh: Mesh object containing vertices and faces
#     loc_dom_band: Array of local dominant bands for each vertex
#     band_idx: Index of the band to analyze (e.g., 4 for B4)
    
#     Returns:
#     float: Number of vertices where this band is dominant
#     float: Percentage of total vertices
#     float: Surface area covered by the band in mm²
#     float: Percentage of total surface area
#     """
#     # Count vertices where this band is dominant
#     band_vertices = loc_dom_band == band_idx
#     num_vertices = np.sum(band_vertices)
#     vertex_percentage = (num_vertices / len(loc_dom_band)) * 100
    
#     # Calculate surface area coverage
#     total_area = 0
#     faces = mesh.faces
#     vertices = mesh.vertices
    
#     for face in faces:
#         # If any vertex in the face has this dominant band, include face area
#         if any(band_vertices[face]):
#             v1, v2, v3 = vertices[face]
#             # Calculate face area using cross product
#             area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
#             total_area += area
    
#     area_percentage = (total_area / mesh.area) * 100
#     return num_vertices, vertex_percentage, total_area, area_percentage

# def process_single_file(filename, surface_path, df):
#     """
#     Process a single surface file and compute various metrics.
#     """
#     try:
#         start_time = time.time()
#         print("Starting processing of {}".format(filename))

#         hemisphere = 'left' if filename.endswith('left.surf.gii') else 'right'
#         participant_session = filename.split('_')[0] + '_' + filename.split('_')[1] + f'_{hemisphere}'
#         base_participant_session = filename.split('_')[0] + '_' + filename.split('_')[1]
        
#         # Get corresponding gestational age
#         try:
#             gestational_age = df[df['participant_session'] == base_participant_session]['scan_age'].values[0]
#         except:
#             print(f"Warning: No matching gestational age found for {base_participant_session}")
#             return None

#         mesh_file = os.path.join(surface_path, filename)
#         if not os.path.exists(mesh_file):
#             print("Error: Mesh file not found: {}".format(mesh_file))
#             return None
            
#         mesh = sio.load_mesh(mesh_file)
#         mesh_smooth_5 = mesh
#         mesh = mesh_smooth_5
#         mesh_save_path = "/scratch/hdienye/mesh_no_curvature/dhcp_full_info/mesh/"
#         ensure_dir_exists(mesh_save_path)
#         new_mesh_path = os.path.join(mesh_save_path, 'smooth_5_{}'.format(filename))
#         sio.write_mesh(mesh, new_mesh_path)
#         N = 5000
        
#         # Compute eigenpairs and mass matrix
#         print("compute the eigen vectors and eigen values")
#         eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

#         # CURVATURE
#         print("compute the mean curvature")
#         PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
#             scurv.curvatures_and_derivatives(mesh)
#         tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
#         principal_tex_dir = '/scratch/hdienye/mesh_no_curvature/dhcp_full_info/principal_curv_tex/'
#         ensure_dir_exists(principal_tex_dir)
#         principal_tex_path = os.path.join(principal_tex_dir, 'principal_curv_{}.gii'.format(filename))
#         sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
        
#         mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
#         tex_mean_curv = stex.TextureND(mean_curv)
#         tex_mean_curv.z_score_filtering(z_thresh=3)
        
#         mean_tex_dir = '/scratch/hdienye/mesh_no_curvature/dhcp_full_info/mean_curv_tex/'
#         ensure_dir_exists(mean_tex_dir)
#         mean_tex_path = os.path.join(mean_tex_dir, 'filt_mean_curv_{}.gii'.format(filename))
#         sio.write_texture(tex_mean_curv, mean_tex_path)
#         filt_mean_curv = tex_mean_curv.darray.squeeze()
#         total_mean_curv = sum(filt_mean_curv)
#         # gyral_mask = np.where(filt_mean_curv > 0, 0, filt_mean_curv) # To mask gyri and only focus on sulci

#         # WHOLE BRAIN MEAN-CURVATURE SPECTRUM
#         grouped_spectrum, group_indices, coefficients, nlevels \
#             = spgy.spectrum(filt_mean_curv, lap_b, eigVects, eigVal)
#         levels = len(group_indices)

#         # Calculate additional measures
#         # 1. Power distribution across bands (normalized to percentage)
#         total_power = np.sum(grouped_spectrum)
#         power_distribution = (grouped_spectrum / total_power) * 100 if total_power > 0 else np.zeros_like(grouped_spectrum)
        
#         # 2. Band wavelength for each band
#         band_wavelengths = calculate_band_wavelength(eigVal, group_indices)
        
#         # Plot coefficients and bands for all mean curvature signal
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
#         frequency = np.sqrt(eigVal/2*np.pi) # from equations in Appendix A.1
#         ax1.scatter(frequency,
#         coefficients, marker='o', s=20, linewidths=0.5)
#         ax1.set_xlabel('Frequency (m⁻¹)')
#         ax1.set_ylabel('Coefficients')

#         ax2.scatter(frequency[1:],
#         coefficients[1:], marker='o', s=20, linewidths=0.5) # remove B0 coefficients
#         ax2.set_xlabel('Frequency (m⁻¹)')
#         ax2.set_ylabel('Coefficients')

#         ax3.bar(np.arange(0, levels), grouped_spectrum)
#         ax3.set_xlabel('Spangy Frequency Bands')
#         ax3.set_ylabel('Power Spectrum')
#         plt.tight_layout()  # Adjust the spacing between subplots
        
#         # Ensure plots directory exists
#         plots_dir = '/scratch/hdienye/mesh_no_curvature/dhcp_full_info/spangy/plots/'
#         ensure_dir_exists(plots_dir)
        
#         # Save first plot
#         fig.savefig(f'{plots_dir}/{filename}.png', bbox_inches='tight', dpi=300)
#         plt.close(fig)
        
#         # Add a second plot for power distribution
#         fig, ax = plt.subplots(figsize=(10, 6))
#         band_names = [f'B{i}' for i in range(levels)]
#         ax.bar(band_names, power_distribution)
#         ax.set_xlabel('Frequency Bands')
#         ax.set_ylabel('Power Distribution (%)')
#         ax.set_title(f'SPANGY Power Distribution - {participant_session}')
#         for i, v in enumerate(power_distribution):
#             ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
#         plt.tight_layout()
#         fig.savefig(f'{plots_dir}/power_distribution_{filename}.png', bbox_inches='tight', dpi=300)
#         plt.close(fig)
        
#         # a. Whole brain parameters
#         mL_in_MM3 = 1000
#         CM2_in_MM2 = 100
#         volume = mesh.volume
#         surface_area = mesh.area
#         afp = np.sum(grouped_spectrum[1:])
#         print('** a. Whole brain parameters **')
#         print('Volume = %d mL, Area = %d cm², Analyze Folding Power = %f,' %
#             (np.floor(volume / mL_in_MM3), np.floor(surface_area / CM2_in_MM2), afp))

#         # LOCAL SPECTRAL BANDS
#         loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, filt_mean_curv,
#                                                             levels, group_indices,
#                                                             eigVects)
        
#         # 3. Calculate number of parcels per band
#         parcels_per_band = calculate_parcels_per_band(loc_dom_band, levels)
        
#         # Print band number of parcels
#         print('** b. Band number of parcels **')
#         for i in range(levels):
#             print(f'B{i} = {parcels_per_band[i]}', end=', ')
#         print()
        
#         # Print band wavelengths
#         print('** Band wavelengths (mm) **')
#         for i in range(levels):
#             print(f'B{i} = {band_wavelengths[i]:.2f}', end=', ')
#         print()

#         # c. Band power
#         print('** c. Band power **')
#         for i in range(levels):
#             print(f'B{i} = {grouped_spectrum[i]:.6f}', end=', ')
#         print()

#         # d. Band relative power
#         print('** d. Band relative power **')
#         for i in range(levels):
#             print(f'B{i} = {power_distribution[i]:.2f}%', end=', ')
#         print()

#         tex_path = f"/scratch/hdienye/mesh_no_curvature/dhcp_full_info/spangy/textures/spangy_dom_band_{participant_session}.gii"
#         tmp_tex = stex.TextureND(loc_dom_band)
#         # tmp_tex.z_score_filtering(z_thresh=3)

#         # Ensure output directories exist before saving files
#         ensure_dir_exists(os.path.dirname(tex_path))
#         sio.write_texture(tmp_tex, tex_path)
        


#         # Save frecomposed data
#         # Create base directory and subdirectories for frecomposed data
#         frecomposed_dir = "/scratch/hdienye/mesh_no_curvature/dhcp_full_info/frecomposed/"
#         bands_dir = os.path.join(frecomposed_dir, "bands")
#         full_dir = os.path.join(frecomposed_dir, "full")
        
#         # Create all directories, ensuring they exist
#         os.makedirs(frecomposed_dir, exist_ok=True)
#         os.makedirs(bands_dir, exist_ok=True)
#         os.makedirs(full_dir, exist_ok=True)
        
#         # Convert each band of frecomposed to a texture and save it
#         # for i in range(frecomposed.shape[1]):
#         #     band_data = frecomposed[:, i]
#         #     band_tex = stex.TextureND(band_data)
#         #     band_path = os.path.join(bands_dir, f'frecomposed_band{i+1}_{filename}.gii')
#         #     sio.write_texture(band_tex, band_path)
            
#         # Also save the full frecomposed array as a numpy file for future analysis
#         np_path = os.path.join(full_dir, f'frecomposed_full_{filename}.npy')
#         np.save(np_path, frecomposed)
        
#         # Get hull area and gyrification index
#         gyrification_index, hull_area = get_gyrification_index(mesh)

#             # Calculate coverage for bands 4, 5, 6 and 7
#         band_vertices = []
#         band_vertex_percentages = []
#         band_areas = []
#         band_area_percentages = []
        
#         max_band = np.max(loc_dom_band)  # Get the highest available band
        
#         for band_idx in [4, 5, 6]:
#             if band_idx <= max_band:
#                 num_verts, vert_pct, area, area_pct = calculate_band_coverage(mesh, loc_dom_band, band_idx)
#             else:
#                 # Set zeros for unavailable bands
#                 num_verts, vert_pct, area, area_pct = 0, 0, 0, 0
            
#             band_vertices.append(num_verts)
#             band_vertex_percentages.append(vert_pct)
#             band_areas.append(area)
#             band_area_percentages.append(area_pct)
        
#         band_powers = []
#         band_rel_powers = []        
#         for band_idx in [4, 5, 6]:
#             if band_idx < len(grouped_spectrum):
#                 band_powers.append(grouped_spectrum[band_idx])
#                 band_rel_powers.append(grouped_spectrum[band_idx]/afp)
#             else:
#                 band_powers.append(0)
#                 band_rel_powers.append(0)

#         # Calculate and print execution time for this iteration
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print(f"\nExecution time for {filename}: {execution_time:.2f} seconds")
#         print("----------------------------------------\n")
        
#         # Create result dictionary with all the requested measures
#         result = {
#             'participant_session': participant_session,
#             'gestational_age': gestational_age,
#             'total_mean_curvature': total_mean_curv,
#             'gyrification_index': gyrification_index,
#             'hull_area': hull_area,
#             'volume_ml': np.floor(volume / mL_in_MM3),
#             'surface_area_cm2': np.floor(surface_area / CM2_in_MM2),
#             'analyze_folding_power': afp,
#             'processing_time': execution_time,
#             "B4_band_relative_power" : band_rel_powers[0],
#             "B5_band_relative_power": band_rel_powers[1], 
#             "B6_band_relative_power": band_rel_powers[2],
#             "B4_number_of_vertices" : band_vertices[0],
#             "B5_number_of_vertices": band_vertices[1], 
#             "B5_number_of_vertices" : band_vertices[2], 
#             # f.write(f"B7: {band_vertices[3]} vertices ({band_vertex_percentages[3]:.2f}%)\n")
#             "B4_vertex_percentage" : band_vertex_percentages[0],
#             "B5_vertex_percentage" : band_vertex_percentages[1],
#             "B6_vertex_percentage" : band_vertex_percentages[2], 
#             "B4_surface_area": band_areas[0],
#             "B5_surface_area": band_areas[1], 
#             "B6_surface_area": band_areas[2], 
#             "B4_surface_area_percentage": band_area_percentages[0],
#             "B5_surface_area_percentage": band_area_percentages[1],
#             "B6_surface_area_percentage": band_area_percentages[2]
#         }
        
#         # Add band power data
#         for i in range(levels):
#             result[f'band_power_B{i}'] = grouped_spectrum[i]
#             result[f'band_power_pct_B{i}'] = power_distribution[i]
#             result[f'band_wavelength_B{i}'] = band_wavelengths[i]
#             result[f'band_parcels_B{i}'] = parcels_per_band[i]
        
#         return result
        
#     except Exception as e:
#         print("Error processing {}: {}".format(filename, str(e)))
#         import traceback
#         traceback.print_exc()
#         return None

# def ensure_dir_exists(directory):
#     """
#     Make sure a directory exists, creating it if necessary
#     Handle the case where the directory might be created between check and creation
#     """
#     try:
#         if not os.path.exists(directory):
#             print(f"Creating directory: {directory}")
#             os.makedirs(directory, exist_ok=True)
#     except FileExistsError:
#         # Directory already exists (race condition handled)
#         print(f"Directory already exists: {directory}")
#         pass

# def main():
#     try:
#         # Paths
#         surface_path = "/scratch/hdienye/mesh_no_curvature/"
#         mesh_info_path = "/scratch/hdienye/participants.tsv"
        
#         # Ensure all output directories exist
#         ensure_dir_exists('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/principal_curv_tex/')
#         ensure_dir_exists('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/mean_curv_tex/')
#         ensure_dir_exists('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/spangy/plots/')
#         ensure_dir_exists('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/spangy/textures/')
#         ensure_dir_exists('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/info/')
        
#         print("Reading data from {}".format(mesh_info_path))
#         # Read dataframe
#         df = pd.read_csv(mesh_info_path, sep='\t')
#         df['participant_session'] = df['subject_id'] + '_' + df['session_id']
        
#         print("Scanning directory: {}".format(surface_path))
#         # Get list of files
#         all_files = [f for f in os.listdir(surface_path) if f.endswith('left.surf.gii') or f.endswith('right.surf.gii')]
#         print("Found {} files to process".format(len(all_files)))
        
#         # Process directly if not in SLURM environment
#         if 'SLURM_ARRAY_TASK_ID' not in os.environ:
#             print("Running in local mode - processing all files")
#             results = []
#             for filename in all_files:
#                 result = process_single_file(filename, surface_path, df)
#                 if result:
#                     results.append(result)
                    
#             if results:
#                 results_df = pd.DataFrame(results)
#                 output_file = os.path.join('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/info/', 'all_results.csv')
#                 results_df.to_csv(output_file, index=False)
#                 print(f"All results saved to {output_file}")
#             else:
#                 print("Warning: No results generated")
#             return
                
#         # Calculate chunk for this array task
#         task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
#         n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
#         chunk_size = len(all_files) // n_tasks + (1 if len(all_files) % n_tasks > 0 else 0)
#         start_idx = task_id * chunk_size
#         end_idx = min((task_id + 1) * chunk_size, len(all_files))
        
#         print("Processing chunk {}/{} (files {} to {})".format(task_id + 1, n_tasks, start_idx, end_idx))
        
#         # Process files in this chunk
#         results = []
#         for filename in all_files[start_idx:end_idx]:
#             result = process_single_file(filename, surface_path, df)
#             if result:
#                 results.append(result)
        
#         # Save results for this chunk
#         if results:
#             results_df = pd.DataFrame(results)
#             chunk_file = os.path.join('/scratch/hdienye/mesh_no_curvature/dhcp_full_info/info/', 'chunk_{}_results.csv'.format(task_id))
#             results_df.to_csv(chunk_file, index=False)
#             print("Results for chunk {} saved to {}".format(task_id, chunk_file))
#         else:
#             print("Warning: No results generated for chunk {}".format(task_id))
            
#     except Exception as e:
#         print("Critical error in main: {}".format(str(e)))
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

# import pandas as pd

# df_1 = pd.read_csv('/home/INT/dienye.h/python_files/combined_dataset/dhcp_qc_filtered.csv')
# df_2 = pd.read_csv('/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv')

# df_1["SITE"] = "dhcp"
# df_2["SITE"] = "marsfet"

# df_combined = pd.concat([df_1, df_2])

# output_file = '/home/INT/dienye.h/python_files/combined_dataset/combined_qc_dataset_with_site.csv'
# df_combined.to_csv(output_file, index=False)



# import pandas as pd

# df = pd.read_csv('/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/harmonized_data_dhcp_marsfet.csv')

# # Keep columns ending with "_harm" or specific columns
# keep_columns = ["gestational_age", "participant_session", "batch"]
# columns_to_keep = [col for col in df.columns if col.endswith('_harm') or col in keep_columns]

# df_filtered = df[columns_to_keep]
# df_filtered.to_csv('/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/dhcp_marsfet_harmonized_params_only.csv', index=False)

# import pandas as pd

# def analyze_band_powers(input_csv_path, output_csv_path):
#     # 1. Load the CSV file
#     df = pd.read_csv(input_csv_path)
    
#     # 2. Check if the required columns exist to prevent errors
#     required_columns = [
#         'band_power_B4_harm', 
#         'band_power_B5_harm', 
#         'band_power_B6_harm', 
#         'analyze_folding_power_harm'
#     ]
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(f"Missing required column: {col}")

#     # 3. Calculate what percentage of the analyzed folding power each band power takes
#     df['pct_B4'] = (df['band_power_B4_harm'] / df['analyze_folding_power_harm']) * 100
#     df['pct_B5'] = (df['band_power_B5_harm'] / df['analyze_folding_power_harm']) * 100
#     df['pct_B6'] = (df['band_power_B6_harm'] / df['analyze_folding_power_harm']) * 100

#     # 4. Calculate the percentage that the sum of all 3 band powers takes
#     df['pct_sum_all_3'] = df['pct_B4'] + df['pct_B5'] + df['pct_B6']
    
#     # 5. Calculate the mean of all the values in the summed percentages column
#     mean_of_summed_percentages = df['pct_sum_all_3'].mean()
    
#     # --- Displaying the results ---
#     print("=== First 5 rows of the calculated percentages ===")
#     print(df[['pct_B4', 'pct_B5', 'pct_B6', 'pct_sum_all_3']].head())
    
#     print("\n=== Final Result ===")
#     print(f"Mean of the summed percentages (B4 + B5 + B6): {mean_of_summed_percentages:.2f}%")
    
#     # 6. Save the new dataframe with the calculated columns to a new CSV
#     df.to_csv(output_csv_path, index=False)
#     print(f"\nSuccessfully saved the detailed results to: {output_csv_path}")
    
#     return mean_of_summed_percentages

# # ==========================================
# # RUN THE CODE
# # Replace these strings with your actual file paths
# # ==========================================
# if __name__ == "__main__":
#     INPUT_CSV = "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/dhcp_marsfet_harmonized_params_only.csv"   # The file you are reading from
#     OUTPUT_CSV = "/home/INT/dienye.h/python_files/rough/temp/analyzed_results.csv" # The new file that will be created
    
#     try:
#         analyze_band_powers(INPUT_CSV, OUTPUT_CSV)
#     except FileNotFoundError:
#         print(f"Error: Could not find the file '{INPUT_CSV}'. Please check the file path.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# import pandas as pd

# def filter_qc_scores(input_csv, output_csv):
#     # 1. Load the CSV file
#     try:
#         df = pd.read_csv(input_csv)
#     except FileNotFoundError:
#         print(f"Error: The file '{input_csv}' was not found.")
#         return
        
#     # 2. Check if the 'qc_score' column exists to prevent errors
#     if 'qc_score' not in df.columns:
#         print("Error: The column 'qc_score' was not found in the CSV file.")
#         print(f"Available columns are: {', '.join(df.columns)}")
#         return

#     # Store the original number of rows for the summary
#     original_row_count = len(df)

#     # 3. Filter the dataframe to keep only rows where qc_score is strictly greater than 2
#     # If you also want to remove empty/NaN qc_scores, you can chain .dropna(subset=['qc_score'])
#     filtered_df = df[df['qc_score'] > 2]

#     # Calculate how many rows were removed
#     filtered_row_count = len(filtered_df)
#     removed_rows = original_row_count - filtered_row_count

#     # 4. Save the filtered dataframe to a new CSV file
#     filtered_df.to_csv(output_csv, index=False)
    
#     # Print a summary of the operation
#     print(f"Filtering complete!")
#     print(f"Original row count: {original_row_count}")
#     print(f"Rows removed (qc_score <= 2): {removed_rows}")
#     print(f"Remaining row count: {filtered_row_count}")
#     print(f"Saved cleaned data to: '{output_csv}'")

# if __name__ == "__main__":
#     # ==========================================
#     # SPECIFY YOUR INPUT AND OUTPUT PATHS HERE
#     # ==========================================
    
#     INPUT_FILE = "/home/INT/dienye.h/Téléchargements/BCN_spangy_fet/spangy_fet/qc_bcn_combined.csv"       # Replace with your actual input file name
#     OUTPUT_FILE = "/home/INT/dienye.h/Téléchargements/BCN_spangy_fet/spangy_fet/qc_filtered_bcn_combined.csv"   # Replace with your desired output file name
    
#     # ==========================================
    
#     filter_qc_scores(INPUT_FILE, OUTPUT_FILE)

# import pandas as pd

# def combine_and_clean_data(file1_path, file2_path, output_path):
#     try:
#         # 1. Load both CSV files
#         print(f"Loading {file1_path}...")
#         df1 = pd.read_csv(file1_path)
        
#         print(f"Loading {file2_path}...")
#         df2 = pd.read_csv(file2_path)
        
#         # 2. Concatenate (stack) the DataFrames vertically
#         # ignore_index=True ensures the row numbers are reset continuously from 0 to the end
#         combined_df = pd.concat([df1, df2], ignore_index=True)
#         print(f"Combined data has {len(combined_df)} rows.")
        
#         # 3. Drop the 'qc_score' column
#         # errors='ignore' prevents the script from crashing if the column is missing
#         if 'qc_score' in combined_df.columns:
#             combined_df = combined_df.drop(columns=['qc_score'])
#             print("Successfully dropped 'qc_score' column.")
#         else:
#             print("Notice: 'qc_score' column was not found in the combined data.")
            
#         # 4. Save the combined and cleaned data to a new CSV
#         combined_df.to_csv(output_path, index=False)
#         print(f"Saved the final dataset to: {output_path}")

#     except FileNotFoundError as e:
#         print(f"Error: Could not find one of the files. Details: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     # ==========================================
#     # SPECIFY YOUR INPUT AND OUTPUT PATHS HERE
#     # ==========================================
    
#     FILE_1 = "/home/INT/dienye.h/python_files/combined_dataset/bcn_chuv_filt_combined.csv"      # Replace with your first file's name
#     FILE_2 = "/home/INT/dienye.h/python_files/combined_dataset/combined_qc_dataset_with_site.csv"     # Replace with your second file's name
#     OUTPUT_FILE = "/home/INT/dienye.h/python_files/combined_dataset/all_sites_filt_combined.csv" # The new combined file to be created
    
#     # ==========================================
    
#     combine_and_clean_data(FILE_1, FILE_2, OUTPUT_FILE)

import pandas as pd
import os

def clean_harmonized_data(input_csv, output_dir):
    try:
        print(f"Loading dataset: {input_csv}")
        df = pd.read_csv(input_csv)
        
        # 1. Find all columns that end with '_harm'
        harm_cols = [col for col in df.columns if col.endswith('_harm')]
        
        if not harm_cols:
            print("Error: No columns ending in '_harm' were found in this file.")
            return

        # 2. Define the exact metadata columns you want to keep
        desired_metadata = ['participant_session', 'gestational_age', 'batch']
        
        # Safely check which of those metadata columns actually exist in the CSV
        actual_metadata = [col for col in desired_metadata if col in df.columns]
        
        if len(actual_metadata) < len(desired_metadata):
            missing = set(desired_metadata) - set(actual_metadata)
            print(f"Warning: Could not find these metadata columns: {missing}")

        # 3. Create the new dataframe with metadata first, then the harmonized columns
        df_cleaned = df[actual_metadata + harm_cols].copy()
        
        # 4. Rename columns by dropping the '_harm' suffix
        # This will only affect the brain metrics; metadata columns remain untouched
        df_cleaned.rename(columns=lambda x: x.replace('_harm', ''), inplace=True)
        
        # 5. Define output path and save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "harmonized_log_transform_all_sites_harm_parms_only.csv")
        
        df_cleaned.to_csv(output_file, index=False)
        print(f"\nSuccess! Kept {len(actual_metadata)} metadata columns and {len(harm_cols)} brain metric columns.")
        print(f"Saved cleaned file to: {output_file}")
        
        # Print a quick peek at the first few rows to verify
        print("\nFirst 5 columns of the cleaned data:")
        print(df_cleaned.head(0)) 

    except FileNotFoundError:
        print(f"Error: Could not find {input_csv}. Please check the path.")

if __name__ == "__main__":
    # ==========================================
    # FILE PATHS
    # ==========================================
    INPUT_FILE = "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/harmonized_log_trans_all_sites.csv"
    OUTPUT_DIR = "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/"
    
    clean_harmonized_data(INPUT_FILE, OUTPUT_DIR)