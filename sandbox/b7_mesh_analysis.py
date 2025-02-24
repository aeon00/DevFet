import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import slam.io as sio
import slam.curvature as scurv
import slam.spangy as spgy
import os
import slam.texture as stex
import time
import sys

def calculate_band_coverage(mesh, loc_dom_band, band_idx):
    """
    Calculate the coverage metrics for a specific frequency band using local dominance map.
    
    Parameters:
    mesh: Mesh object containing vertices and faces
    loc_dom_band: Array of local dominant bands for each vertex
    band_idx: Index of the band to analyze (e.g., 4 for B4)
    
    Returns:
    float: Number of vertices where this band is dominant
    float: Percentage of total vertices
    float: Surface area covered by the band in mm²
    float: Percentage of total surface area
    """
    # Count vertices where this band is dominant
    band_vertices = loc_dom_band == band_idx
    num_vertices = np.sum(band_vertices)
    vertex_percentage = (num_vertices / len(loc_dom_band)) * 100
    
    # Calculate surface area coverage
    total_area = 0
    faces = mesh.faces
    vertices = mesh.vertices
    
    for face in faces:
        # If any vertex in the face has this dominant band, include face area
        if any(band_vertices[face]):
            v1, v2, v3 = vertices[face]
            # Calculate face area using cross product
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            total_area += area
    
    area_percentage = (total_area / mesh.area) * 100
    return num_vertices, vertex_percentage, total_area, area_percentage

def save_results(N, volume, total_surface_area, afp, band_powers, band_rel_powers, 
                band_vertices, band_vertex_percentages,
                band_areas, band_area_percentages, processing_time,
                output_file="folding_results.txt"):
    # Calculate total coverage as sanity check
    total_covered_area = sum(band_areas)
    coverage_ratio = total_covered_area / total_surface_area
    with open(output_file, "a") as f:
        f.write(f"N={N}\n")
        f.write(f"Volume={np.floor(volume/1000)} mL\n")
        f.write(f"Total Surface Area={np.floor(total_surface_area/100)} cm²\n")
        f.write(f"Total Area Covered by Bands={np.floor(total_covered_area/100)} cm²\n")
        f.write(f"Coverage Ratio={coverage_ratio:.2f}\n")
        f.write(f"Analyze Folding Power={afp}\n")
        f.write(f"Band Powers: B4={band_powers[0]}, B5={band_powers[1]}, B6={band_powers[2]}, B7={band_powers[3]}\n")
        f.write(f"Band Relative Powers: B4={band_rel_powers[0]:.5f}, B5={band_rel_powers[1]:.5f}, B6={band_rel_powers[2]:.5f}, B7={band_rel_powers[3]:.5f}\n")
        f.write("\nBand Coverage Analysis:\n")
        f.write("\nVertex Coverage:\n")
        f.write(f"B4: {band_vertices[0]} vertices ({band_vertex_percentages[0]:.2f}%)\n")
        f.write(f"B5: {band_vertices[1]} vertices ({band_vertex_percentages[1]:.2f}%)\n")
        f.write(f"B6: {band_vertices[2]} vertices ({band_vertex_percentages[2]:.2f}%)\n")
        f.write(f"B7: {band_vertices[3]} vertices ({band_vertex_percentages[3]:.2f}%)\n")
        f.write("\nSurface Area Coverage:\n")
        f.write(f"B4: {band_areas[0]:.2f} mm² ({band_area_percentages[0]:.2f}%)\n")
        f.write(f"B5: {band_areas[1]:.2f} mm² ({band_area_percentages[1]:.2f}%)\n")
        f.write(f"B6: {band_areas[2]:.2f} mm² ({band_area_percentages[2]:.2f}%)\n")
        f.write(f"B7: {band_areas[3]:.2f} mm² ({band_area_percentages[3]:.2f}%)\n")
        f.write("\nProcessing Time:\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds")
        f.write("-" * 50 + "\n")

def process_single_file(mesh_path):
    # Initialize lists to store metrics for plotting
    N_values = []
    surface_areas = {'B4': [], 'B5': [], 'B6': [], 'B7': []}
    area_percentages = {'B4': [], 'B5': [], 'B6': [], 'B7': []}
    band_powers_list = {'B4': [], 'B5': [], 'B6': [], 'B7': []}

    # Load mesh'


    for i in os.listdir(mesh_path):
        start_time = time.time()
        N = 7000
        mesh = os.path.join(mesh_path, i)
        mesh = sio.load_mesh(mesh)
        vertices = mesh.vertices
        num_vertices = len(vertices)

        # Calculate eigenpairs
        eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)
        
        # Calculate curvatures
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        tex_mean_curv = stex.TextureND(mean_curv)
        tex_mean_curv.z_score_filtering(z_thresh=3)
        mean_curv = tex_mean_curv.darray.squeeze() #filtered mean curvature
        
        # Calculate spectrum
        grouped_spectrum, group_indices, coefficients, nlevels = spgy.spectrum(mean_curv, lap_b,
                                                                            eigVects, eigVal)
        
        # Calculate local dominance map
        loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, mean_curv,
                                                            nlevels, group_indices,
                                                            eigVects)
        
        tex_path = f"/envau/work/meca/users/dienye.h/B7_analysis/textures/spangy_dom_band_{i}.gii"
        tmp_tex = stex.TextureND(loc_dom_band)
        # tmp_tex.z_score_filtering(z_thresh=3)
        sio.write_texture(tmp_tex, tex_path)
        
        # Calculate basic metrics
        mL_in_MM3 = 1000
        CM2_in_MM2 = 100
        volume = mesh.volume
        surface_area = mesh.area
        afp = np.sum(grouped_spectrum[1:])
        
        # Handle cases where B6 might not be available
        band_powers = []
        band_rel_powers = []
        for band_idx in [4, 5, 6, 7]:
            if band_idx < len(grouped_spectrum):
                band_powers.append(grouped_spectrum[band_idx])
                band_rel_powers.append(grouped_spectrum[band_idx]/afp)
            else:
                band_powers.append(0)
                band_rel_powers.append(0)
        
        # Calculate coverage for bands 4, 5, 6 and 7
        band_vertices = []
        band_vertex_percentages = []
        band_areas = []
        band_area_percentages = []
        
        max_band = np.max(loc_dom_band)  # Get the highest available band
        
        for band_idx in [4, 5, 6, 7]:
            if band_idx <= max_band:
                num_verts, vert_pct, area, area_pct = calculate_band_coverage(mesh, loc_dom_band, band_idx)
            else:
                # Set zeros for unavailable bands
                num_verts, vert_pct, area, area_pct = 0, 0, 0, 0
            
            band_vertices.append(num_verts)
            band_vertex_percentages.append(vert_pct)
            band_areas.append(area)
            band_area_percentages.append(area_pct)
        
        end_time = time.time()
        execution_time = end_time - start_time

        # Save results
        output_file = f'spectrum_results_{i}.txt'
        output_path = "/envau/work/meca/users/dienye.h/B7_analysis/"
        output_file_dir = os.path.join(output_path, output_file)
        save_results(N, volume, surface_area, afp, band_powers, band_rel_powers,
                    band_vertices, band_vertex_percentages,
                    band_areas, band_area_percentages, execution_time, output_file_dir)


def main():
    try:
        # Paths
        mesh_path = '/envau/work/meca/users/dienye.h/B7_analysis/mesh/'
        
        print("Scanning directory: {}".format(mesh_path))
        # Get list of files
        all_files = [f for f in os.listdir(mesh_path) if f.endswith('left.surf.gii') or f.endswith('right.surf.gii')]
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
            mesh_file = os.path.join(mesh_path, filename)
            result = process_single_file(mesh_file)
            
    except Exception as e:
        print("Critical error in main: {}".format(str(e)))
        sys.exit(1)

if __name__ == "__main__":
    main()