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
    Process a single surface file and compute various metrics.
    """
    try:
        # Define output paths that will be used to check if this file was already processed
        smooth_output_path = os.path.join('/scratch/hdienye/dhcp_full_info/inflated_mesh', 'inflated_{}.gii'.format(str(filename)))
        
        # Check if output files already exist, indicating this file was already processed
        if os.path.exists(smooth_output_path):
            print(f"Skipping {filename} as it was already processed.")
            return None
            
        start_time = time.time()
        print("Starting processing of {}".format(filename))
    
        mesh_file = os.path.join(surface_path, filename)
        if not os.path.exists(mesh_file):
            print("Error: Mesh file not found: {}".format(mesh_file))
            return None
            
        mesh = sio.load_mesh(mesh_file)
        mesh = laplacian_mesh_smoothing(mesh, nb_iter=5, dt=10)
        filename = str(filename)
        new_mesh_path = smooth_output_path
        sio.write_mesh(mesh, new_mesh_path)
        
    except Exception as e:
        print("Error processing {}: {}".format(filename, str(e)))
        return None

def main():
    try:
        # Paths
        surface_path = "/scratch/gauzias/data/datasets/dhcp_fetal_bids/output/svrtk_BOUNTI/output_BOUNTI_surfaces/"
        mesh_info_path = "/scratch/hdienye/participants.tsv"
        
        print("Reading data from {}".format(mesh_info_path))
        # Read dataframe
        df = pd.read_csv(mesh_info_path, sep='\t')
        df['participant_session'] = df['participant_id'] + '_' + df['session_id']
        
        print("Scanning directory: {}".format(surface_path))
        # Get list of files
        all_files = [f for f in os.listdir(surface_path) if f.endswith('left.surf.gii') or f.endswith('right.surf.gii')]
        print("Found {} files to process".format(len(all_files)))
        
        # Check which files have already been processed
        processed_files = []
        files_to_process = []
        
        for filename in all_files:
            # Define output paths that will be used to check if this file was already processed
            smooth_output_path = os.path.join('/scratch/hdienye/dhcp_full_info/inflated_mesh', 'inflated_{}.gii'.format(str(filename)))
            
            if os.path.exists(smooth_output_path):
                processed_files.append(filename)
            else:
                files_to_process.append(filename)
        
        print(f"Found {len(processed_files)} already processed files and {len(files_to_process)} files to process")
        
        # If all files are processed, exit early
        if not files_to_process:
            print("All files have already been processed. Exiting.")
            return
        
        # Calculate chunk for this array task
        task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        n_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        chunk_size = len(files_to_process) // n_tasks + (1 if len(files_to_process) % n_tasks > 0 else 0)
        start_idx = task_id * chunk_size
        end_idx = min((task_id + 1) * chunk_size, len(files_to_process))
        
        print(f"Processing chunk {task_id + 1}/{n_tasks} (files {start_idx} to {end_idx - 1} of remaining files)")
        
        # Process files in this chunk
        results = []
        for filename in files_to_process[start_idx:end_idx]:
            result = process_single_file(filename, surface_path, df)
            
    except Exception as e:
        print("Critical error in main: {}".format(str(e)))
        sys.exit(1)

if __name__ == "__main__":
    main()
