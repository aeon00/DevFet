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


def process_single_file(filename, surface_path):
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

        # CURVATURE
        print("compute the mean curvature")
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)
        tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
        principal_tex_path = os.path.join('/envau/work/meca/users/dienye.h/Curvature/principal_curv_tex/', 'principal_curv_{}.gii'.format(filename))
        sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        mean_tex_path = os.path.join('/envau/work/meca/users/dienye.h/Curvature/mean_curv_tex/', 'filt_mean_curv_{}.gii'.format(filename))
        sio.write_texture(tex_mean_curv, mean_tex_path)

        
    except Exception as e:
        print("Error processing {}: {}".format(filename, str(e)))
        return None

def main():
    try:
        # Paths
        surface_path = "/envau/work/meca/data/Fetus/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_surfaces/haste/"
        
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
        for filename in all_files[start_idx:end_idx]:
            result = process_single_file(filename, surface_path)
            
    except Exception as e:
        print("Critical error in main: {}".format(str(e)))
        sys.exit(1)

if __name__ == "__main__":
    main()