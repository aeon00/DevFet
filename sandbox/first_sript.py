import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
import os
import pandas as pd
import time
import slam.spangy as spgy
import numpy as np
import matplotlib.pyplot as plt

def process_single_file(filename, surface_path, df):
    try:
        start_time = time.time()
        
        participant_session = filename.split('_')[0] + '_' + filename.split('_')[1]
        
        try:
            gestational_age = df[df['participant_session'] == participant_session]['fetus_gestational_age_at_scan'].values[0]
        except:
            print(f"Warning: No matching gestational age found for {participant_session}")
            return None

        mesh_file = os.path.join(surface_path, filename)
        mesh = sio.load_mesh(mesh_file)
        mesh.apply_transform(mesh.principal_inertia_transform)
        N = 4000
        
        print(f"Computing eigen vectors and eigen values for {filename}")
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

        print(f"Computing mean curvature for {filename}")
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        total_mean_curv = sum(mean_curv)

        grouped_spectrum, group_indices, coefficients, nlevels \
            = spgy.spectrum(mean_curv, lap_b, eigVects, eigVal)
        levels = len(group_indices)

        # Create and save plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        frequency = np.sqrt(eigVal/2*np.pi)
        ax1.scatter(frequency, coefficients, marker='o', s=20, linewidths=0.5)
        ax1.set_xlabel('Frequency (m⁻¹)')
        ax1.set_ylabel('Coefficients')

        ax2.scatter(frequency[1:], coefficients[1:], marker='o', s=20, linewidths=0.5)
        ax2.set_xlabel('Frequency (m⁻¹)')
        ax2.set_ylabel('Coefficients')

        ax3.bar(np.arange(0, levels), grouped_spectrum)
        ax3.set_xlabel('Spangy Frequency Bands')
        ax3.set_ylabel('Power Spectrum')
        plt.tight_layout()
        fig.savefig(f'/scratch/hdienye/spangy/plots/plot_{filename}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Calculate metrics
        mL_in_MM3 = 1000
        CM2_in_MM2 = 100
        volume = mesh.volume
        surface_area = mesh.area
        afp = np.sum(grouped_spectrum[1:])

        # Process local spectral bands
        loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, mean_curv,
                                                            levels, group_indices,
                                                            eigVects)

        tex_path = f"/scratch/hdienye/spangy/textures/spangy_dom_band_{filename}.gii"
        tmp_tex = stex.TextureND(loc_dom_band)
        sio.write_texture(tmp_tex, tex_path)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {filename}: {execution_time:.2f} seconds")
        
        return {
            'participant_session': participant_session,
            'gestational_age': gestational_age,
            'total_mean_curvature': total_mean_curv,
            'band_power_B4': grouped_spectrum[4],
            'band_power_B5': grouped_spectrum[5],
            'band_power_B6': grouped_spectrum[6],
            'volume_ml': np.floor(volume / mL_in_MM3),
            'surface_area_cm2': np.floor(surface_area / CM2_in_MM2),
            'analyze_folding_power': afp
        }
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

def main():
    # Paths
    surface_path = "/scratch/gauzias/data/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_surfaces/haste"
    mesh_info_path = "/scratch/gauzias/code_gui/fet-processing/data/tables/marsFet_HASTE_lastest_volumes_BOUNTI.csv"
    
    # Read dataframe
    df = pd.read_csv(mesh_info_path)
    df['participant_session'] = df['participant_id'] + '_' + df['session_id']
    
    # Get list of files
    all_files = [f for f in os.listdir(surface_path) if f.endswith('surf.gii')]
    
    # Calculate chunk for this array task
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
    chunk_size = len(all_files) // n_tasks + (1 if len(all_files) % n_tasks > 0 else 0)
    start_idx = task_id * chunk_size
    end_idx = min((task_id + 1) * chunk_size, len(all_files))
    
    # Process files in this chunk
    results = []
    for filename in all_files[start_idx:end_idx]:
        result = process_single_file(filename, surface_path, df)
        if result:
            results.append(result)
    
    # Save results for this chunk
    if results:
        results_df = pd.DataFrame(results)
        chunk_file = f'/scratch/hdienye/spangy/results/chunk_{task_id}_results.csv'
        results_df.to_csv(chunk_file, index=False)
        print(f"Results for chunk {task_id} saved to {chunk_file}")

if __name__ == "__main__":
    main()