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

def local_determinance_map(coefficients, f2analyse, nlevels, group_indices, eig_vec):
    """
    Parameters
    ----------
    coefficients : Array of floats
        Fourier coefficients of the input function f2analyse
    f2analyse : Array of floats
        Scalar function to analyze (e.g. mean curvature)
    nlevels : Array of ints
        number of spectral bands
    group_indices : Array of ints
        indices of spectral bands
    eig_vec : Array of floats
        eigenvectors (reversed order for computation and memory reasons)

    Returns
    -------
    loc_det_band : Array of floats
        texture with the differential contribution of each frequency band
    frecomposed : Array of floats
        recomposition of f2analyse in each frequency band
    """
    N = np.size(coefficients)
    num_vertices = len(f2analyse)

    # Band by band recomposition
    frecomposed = np.zeros((num_vertices, nlevels - 1), dtype=np.float64)
    eig_vec = np.flip(eig_vec, 1)

    for i in range(nlevels - 1):
        levels_i = np.arange(
            group_indices[i + 1, 0], group_indices[i + 1, 1] + 1)
        f_ii = np.dot(eig_vec[:, N - levels_i - 1], coefficients[levels_i].T)
        frecomposed[:, i] = f_ii

    # Compute cumulative synthesis
    cumulative_synthesis = np.zeros((num_vertices, nlevels))
    for k in range(1, nlevels):
        if k == 1:
            cumulative_synthesis[:, k] = frecomposed[:, k-1]
        else:
            cumulative_synthesis[:, k] = cumulative_synthesis[:, k-1] + frecomposed[:, k-1]

    # Compute SMk according to formula C.2-C.4 in the paper
    SMk = np.zeros((num_vertices, nlevels))
    for k in range(1, nlevels):
        # Formula C.3
        positive_mask = (cumulative_synthesis[:, k] > 0)
        # Formula C.4
        positive_prev_mask = (cumulative_synthesis[:, k-1] > 0) if k > 1 else np.zeros(num_vertices, dtype=bool)
        
        # Formula C.2
        SMk[:, k] = positive_mask.astype(int) - positive_prev_mask.astype(int)

    # Initialize determinant band map
    loc_det_band = np.zeros(num_vertices)

    # Apply the segmentation logic from the paper
    # First assign the highest band that causes a sign change
    for k in range(nlevels-1, 0, -1):  # Start from highest band and work down
        # Sulci (negative sign change)
        sulci_indices = np.where((SMk[:, k] < 0))
        loc_det_band[sulci_indices] = -k
        
        # Gyri (positive sign change)
        gyri_indices = np.where((SMk[:, k] > 0))
        loc_det_band[gyri_indices] = k

    return loc_det_band, frecomposed

def extract_sub_sess_left(filename):
    import re
    
    # Define the pattern to match sub-XXXX_ses-XXXX and add "left" at the end
    pattern = r'(sub-\d+)_(ses-\d+).*?(left)'
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        # Extract the matched groups and combine them
        return f"{match.group(1)}_{match.group(2)}_left"
    else:
        return None

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
        f.write(f"Band Powers: B4={band_powers[0]}, B5={band_powers[1]}, B6={band_powers[2]}")
        f.write(f"Band Relative Powers: B4={band_rel_powers[0]:.5f}, B5={band_rel_powers[1]:.5f}, B6={band_rel_powers[2]:.5f}")
        f.write("\nBand Coverage Analysis:\n")
        f.write("\nVertex Coverage:\n")
        f.write(f"B4: {band_vertices[0]} vertices ({band_vertex_percentages[0]:.2f}%)\n")
        f.write(f"B5: {band_vertices[1]} vertices ({band_vertex_percentages[1]:.2f}%)\n")
        f.write(f"B6: {band_vertices[2]} vertices ({band_vertex_percentages[2]:.2f}%)\n")
        # f.write(f"B7: {band_vertices[3]} vertices ({band_vertex_percentages[3]:.2f}%)\n")
        f.write("\nSurface Area Coverage:\n")
        f.write(f"B4: {band_areas[0]:.2f} mm² ({band_area_percentages[0]:.2f}%)\n")
        f.write(f"B5: {band_areas[1]:.2f} mm² ({band_area_percentages[1]:.2f}%)\n")
        f.write(f"B6: {band_areas[2]:.2f} mm² ({band_area_percentages[2]:.2f}%)\n")
        # f.write(f"B7: {band_areas[3]:.2f} mm² ({band_area_percentages[3]:.2f}%)\n")
        f.write("\nProcessing Time:\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds")
        f.write("-" * 50 + "\n")

def process_single_file(mesh_file):

    # Load mesh'
    start_time = time.time()
    N = 5000
    mesh = sio.load_mesh(mesh_file)
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
    loc_det_band, frecomposed = local_determinance_map(coefficients, mean_curv,
                                                        nlevels, group_indices,
                                                        eigVects)
    
    file = extract_sub_sess_left(mesh_file)

    # frec_tex_path = "/envau/work/meca/users/dienye.h/det_band_test_results/textures/frec_{file}.gii"
    # frec_tex = stex.TextureND(frecomposed)
    # sio.write_texture(frec_tex, frec_tex_path)

    tex_path = f"/envau/work/meca/users/dienye.h/det_band_test_results/textures/spangy_det_band_{file}.gii"
    tmp_tex = stex.TextureND(loc_det_band)
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
    for band_idx in [4, 5, 6]:
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
    
    max_band = np.max(loc_det_band)  # Get the highest available band
    
    for band_idx in [4, 5, 6]:
        if band_idx <= max_band:
            num_verts, vert_pct, area, area_pct = calculate_band_coverage(mesh, loc_det_band, band_idx)
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
    output_file = f'spectrum_results_{file}.txt'
    output_path = "/envau/work/meca/users/dienye.h/det_band_test_results/"
    output_file_dir = os.path.join(output_path, output_file)
    save_results(N, volume, surface_area, afp, band_powers, band_rel_powers,
                band_vertices, band_vertex_percentages,
                band_areas, band_area_percentages, execution_time, output_file_dir)


def main():
    # try:
    #     # Paths
    #     mesh_path = '/home/INT/dienye.h/Python Codes/rough_hemisphere/mesh_surfaces/'
        
    #     print("Scanning directory: {}".format(mesh_path))
    #     # Get list of files
    #     all_files = [f for f in os.listdir(mesh_path) if f.endswith('left.surf.gii') or f.endswith('right.surf.gii')]
    #     print("Found {} files to process".format(len(all_files)))
        
        
    #     # Process files in this chunk
    #     for filename in all_files:
    #         mesh_file = os.path.join(mesh_path, filename)
    #         result = process_single_file(mesh_file)
            
    # except Exception as e:
    #     print("Critical error in main: {}".format(str(e)))
    #     sys.exit(1)

        # Paths
    mesh_path = '/envau/work/meca/users/dienye.h/rough_hemisphere/mesh_surfaces/'
    
    print("Scanning directory: {}".format(mesh_path))
    # Get list of files
    all_files = [f for f in os.listdir(mesh_path)]
    print("Found {} files to process".format(len(all_files)))
    
    
    # Process files in this chunk
    for filename in all_files:
        mesh_file = os.path.join(mesh_path, filename)
        result = process_single_file(mesh_file)

if __name__ == "__main__":
    main()