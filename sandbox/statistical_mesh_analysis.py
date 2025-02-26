# import numpy as np
# import trimesh
# import nibabel as nib
# from sklearn.neighbors import NearestNeighbors
# import time
# import os
# import glob
# import matplotlib.pyplot as plt
# import pandas as pd
# from datetime import datetime

# def load_mesh(mesh_path):
#     """
#     Load a mesh file, supporting GIFTI (.gii) format.
    
#     Parameters:
#     -----------
#     mesh_path : str
#         Path to the mesh file
        
#     Returns:
#     --------
#     vertices : np.array
#         Vertex coordinates
#     faces : np.array
#         Face indices
#     """
#     print(f"Loading mesh from {mesh_path}...")
    
#     # Check if file is a GIFTI file
#     if mesh_path.lower().endswith('.gii'):
#         return load_gifti(mesh_path)
#     else:
#         # Use trimesh for other file types
#         mesh = trimesh.load_mesh(mesh_path)
#         return mesh.vertices, mesh.faces

# def load_gifti(gifti_path):
#     """
#     Load a GIFTI format mesh file.
    
#     Parameters:
#     -----------
#     gifti_path : str
#         Path to the GIFTI file
        
#     Returns:
#     --------
#     vertices : np.array
#         Vertex coordinates
#     faces : np.array
#         Face indices
#     """
#     try:
#         # Load GIFTI file
#         gifti_img = nib.load(gifti_path)
        
#         # Extract vertices and faces from data arrays
#         data_arrays = gifti_img.darrays
        
#         vertices = None
#         faces = None
        
#         # Try to identify arrays by intent
#         for da in data_arrays:
#             if hasattr(da, 'intent') and da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
#                 vertices = da.data
#             elif hasattr(da, 'intent') and da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
#                 faces = da.data
        
#         # If intent-based detection failed, try position-based
#         if vertices is None and len(data_arrays) > 0:
#             vertices = data_arrays[0].data
#         if faces is None and len(data_arrays) > 1:
#             faces = data_arrays[1].data
        
#         # Ensure we have valid data
#         if vertices is not None:
#             vertices = vertices.astype(np.float64)
#         else:
#             raise ValueError("Could not extract vertices from GIFTI file")
            
#         if faces is not None:
#             faces = faces.astype(np.int32)
#         else:
#             raise ValueError("Could not extract faces from GIFTI file")
        
#         return vertices, faces
        
#     except Exception as e:
#         raise ValueError(f"Error loading GIFTI file: {str(e)}")

# def compute_statistical_outliers(vertices, k=30, std_ratio=2.0, batch_size=10000):
#     """
#     Identify statistical outliers based on distance to neighbors.
#     Optimized for memory efficiency and speed.
    
#     Parameters:
#     -----------
#     vertices : np.array
#         Vertex coordinates
#     k : int
#         Number of nearest neighbors to consider
#     std_ratio : float
#         Number of standard deviations to use as threshold
#     batch_size : int
#         Size of batches for processing large meshes
        
#     Returns:
#     --------
#     outlier_scores : np.array
#         Outlier score for each vertex (higher = more likely to be an outlier)
#     is_outlier : np.array
#         Boolean array indicating if vertex is an outlier
#     """
#     start_time = time.time()
#     num_vertices = len(vertices)
    
#     print(f"Computing statistical outliers for {num_vertices} vertices (k={k}, std_ratio={std_ratio})...")
    
#     # For large meshes, limit k to prevent excessive memory usage
#     if num_vertices > 1000000 and k > 20:
#         k = 20
#         print(f"Reduced k to {k} due to large mesh size")
    
#     # Initialize output array
#     mean_distances = np.zeros(num_vertices)
    
#     # Build KD-tree for efficient neighbor searches
#     print("Building KD-tree for nearest neighbor search...")
#     tree = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', leaf_size=40, n_jobs=-1)
#     tree.fit(vertices)
    
#     # Process in batches to manage memory usage
#     num_batches = int(np.ceil(num_vertices / batch_size))
#     print(f"Processing {num_vertices} vertices in {num_batches} batches...")
    
#     for batch_idx in range(num_batches):
#         batch_start = batch_idx * batch_size
#         batch_end = min((batch_idx + 1) * batch_size, num_vertices)
        
#         print(f"Processing batch {batch_idx+1}/{num_batches} (vertices {batch_start}-{batch_end})...")
        
#         # Get vertices for this batch
#         batch_vertices = vertices[batch_start:batch_end]
        
#         # Find k nearest neighbors
#         distances, _ = tree.kneighbors(batch_vertices)
        
#         # Average distance to k nearest neighbors (excluding self)
#         batch_mean_distances = np.mean(distances[:, 1:], axis=1)
        
#         # Store results
#         mean_distances[batch_start:batch_end] = batch_mean_distances
    
#     # Compute global statistics
#     global_mean = np.mean(mean_distances)
#     global_std = np.std(mean_distances)
    
#     # Compute outlier scores
#     outlier_scores = (mean_distances - global_mean) / global_std
#     is_outlier = outlier_scores > std_ratio
    
#     print(f"Statistical outlier computation completed in {time.time() - start_time:.2f} seconds")
#     print(f"Identified {np.sum(is_outlier)} outliers out of {num_vertices} vertices ({np.mean(is_outlier)*100:.2f}%)")
    
#     return outlier_scores, is_outlier

# def visualize_outliers(vertices, faces, outlier_scores, output_path, colormap='viridis'):
#     """
#     Export mesh with outlier visualization.
    
#     Parameters:
#     -----------
#     vertices : np.array
#         Vertex coordinates
#     faces : np.array
#         Face indices
#     outlier_scores : np.array
#         Outlier scores for each vertex
#     output_path : str
#         Path to save the visualization
#     colormap : str
#         Matplotlib colormap to use
#     """
#     print(f"Visualizing outliers using colormap '{colormap}'...")
    
#     # Create temporary mesh for visualization
#     mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
#     # Normalize scores to [0,1]
#     if np.max(outlier_scores) > np.min(outlier_scores):
#         normalized_scores = (outlier_scores - np.min(outlier_scores)) / (np.max(outlier_scores) - np.min(outlier_scores))
#     else:
#         normalized_scores = np.zeros_like(outlier_scores)
    
#     # Get colormap
#     import matplotlib.cm as cm
#     cmap = cm.get_cmap(colormap)
    
#     # Apply colormap (red for high outlier scores)
#     colors = cmap(normalized_scores)[:, 0:3]
    
#     # Create colored mesh
#     colored_mesh = mesh.copy()
#     colored_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
    
#     # Export mesh
#     colored_mesh.export(output_path)
#     print(f"Visualization saved to {output_path}")

# def analyze_outliers(mesh_path, output_dir, k=30, std_ratio=2.0):
#     """
#     Analyze statistical outliers in a mesh and generate results.
    
#     Parameters:
#     -----------
#     mesh_path : str
#         Path to the mesh file
#     output_dir : str
#         Directory to save results
#     k : int
#         Number of nearest neighbors to consider
#     std_ratio : float
#         Number of standard deviations to use as threshold
        
#     Returns:
#     --------
#     result_summary : dict
#         Dictionary with summary statistics
#     """
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load mesh
#     vertices, faces = load_mesh(mesh_path)
#     num_vertices = len(vertices)
    
#     print(f"Loaded mesh with {num_vertices} vertices and {len(faces)} faces")
    
#     # Adjust parameters for very large meshes
#     if num_vertices > 1000000:
#         k = min(k, 20)
#         batch_size = 5000
#     elif num_vertices > 500000:
#         k = min(k, 25)
#         batch_size = 10000
#     else:
#         batch_size = 20000
    
#     # Compute outliers
#     outlier_scores, is_outlier = compute_statistical_outliers(
#         vertices, k=k, std_ratio=std_ratio, batch_size=batch_size
#     )
    
#     # Save results
#     np.savetxt(os.path.join(output_dir, 'outlier_scores.txt'), outlier_scores)
#     np.savetxt(os.path.join(output_dir, 'is_outlier.txt'), is_outlier.astype(int))
    
#     # Create histogram
#     plt.figure(figsize=(10, 6))
#     plt.hist(outlier_scores, bins=100)
#     plt.axvline(x=std_ratio, color='r', linestyle='--', label=f'Threshold ({std_ratio} std)')
#     plt.title('Distribution of Outlier Scores')
#     plt.xlabel('Outlier Score (standard deviations)')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'outlier_histogram.png'))
#     plt.close()
    
#     # Visualize outliers
#     visualize_outliers(
#         vertices, faces, outlier_scores, 
#         os.path.join(output_dir, 'outlier_visualization.ply')
#     )
    
#     # Calculate statistics
#     outlier_count = np.sum(is_outlier)
#     outlier_percentage = np.mean(is_outlier) * 100
#     mean_score = np.mean(outlier_scores)
#     max_score = np.max(outlier_scores)
    
#     # Determine concern level
#     if outlier_percentage > 5:
#         concern_level = "HIGH"
#     elif outlier_percentage > 1:
#         concern_level = "MODERATE"
#     else:
#         concern_level = "LOW"
    
#     # Write summary
#     with open(os.path.join(output_dir, 'outlier_summary.txt'), 'w') as f:
#         f.write("Statistical Outlier Analysis Summary\n")
#         f.write("===================================\n\n")
#         f.write(f"Mesh: {os.path.basename(mesh_path)}\n")
#         f.write(f"Vertices: {num_vertices}\n")
#         f.write(f"Faces: {len(faces)}\n\n")
#         f.write(f"Analysis Parameters:\n")
#         f.write(f"- Nearest neighbors (k): {k}\n")
#         f.write(f"- Outlier threshold: {std_ratio} standard deviations\n\n")
#         f.write(f"Results:\n")
#         f.write(f"- Outlier count: {outlier_count}\n")
#         f.write(f"- Outlier percentage: {outlier_percentage:.2f}%\n")
#         f.write(f"- Mean outlier score: {mean_score:.4f}\n")
#         f.write(f"- Maximum outlier score: {max_score:.4f}\n\n")
#         f.write(f"Assessment: {concern_level} CONCERN - ")
        
#         if concern_level == "HIGH":
#             f.write("Large percentage of statistical outliers detected\n")
#         elif concern_level == "MODERATE":
#             f.write("Notable percentage of statistical outliers detected\n")
#         else:
#             f.write("Small percentage of statistical outliers detected\n")
    
#     print(f"Analysis complete. Results saved to {output_dir}")
    
#     # Return summary for batch processing
#     return {
#         "mesh": os.path.basename(mesh_path),
#         "vertices": num_vertices,
#         "faces": len(faces),
#         "outlier_count": outlier_count,
#         "outlier_percentage": outlier_percentage,
#         "mean_score": mean_score,
#         "max_score": max_score,
#         "concern_level": concern_level
#     }

# def batch_process_directory(input_dir, output_dir, file_pattern="*.gii", k=30, std_ratio=2.0):
#     """
#     Process all mesh files in a directory.
    
#     Parameters:
#     -----------
#     input_dir : str
#         Directory containing mesh files
#     output_dir : str
#         Directory to save results
#     file_pattern : str
#         Pattern to match files (e.g., "*.gii", "*.obj")
#     k : int
#         Number of nearest neighbors to consider
#     std_ratio : float
#         Number of standard deviations to use as threshold
#     """
#     # Create main output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Find all matching files
#     mesh_files = glob.glob(os.path.join(input_dir, file_pattern))
    
#     if not mesh_files:
#         print(f"No files matching pattern '{file_pattern}' found in {input_dir}")
#         return
    
#     print(f"Found {len(mesh_files)} files to process")
    
#     # Create timestamp for this batch run
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Summary results to collect
#     summary_results = []
    
#     # Process each file
#     for i, mesh_path in enumerate(mesh_files):
#         try:
#             mesh_name = os.path.basename(mesh_path)
#             mesh_output_dir = os.path.join(output_dir, os.path.splitext(mesh_name)[0])
            
#             print(f"\n[{i+1}/{len(mesh_files)}] Processing {mesh_name}...")
            
#             # Analyze this mesh
#             result = analyze_outliers(mesh_path, mesh_output_dir, k=k, std_ratio=std_ratio)
            
#             # Add to summary
#             summary_results.append(result)
            
#         except Exception as e:
#             print(f"Error processing {mesh_path}: {e}")
#             # Add error entry to summary
#             summary_results.append({
#                 "mesh": os.path.basename(mesh_path),
#                 "error": str(e)
#             })
    
#     # Create summary report
#     if summary_results:
#         # CSV summary
#         summary_df = pd.DataFrame(summary_results)
#         summary_csv_path = os.path.join(output_dir, f"batch_summary_{timestamp}.csv")
#         summary_df.to_csv(summary_csv_path, index=False)
        
#         # Text summary
#         with open(os.path.join(output_dir, f"batch_summary_{timestamp}.txt"), 'w') as f:
#             f.write("Batch Statistical Outlier Analysis Summary\n")
#             f.write("========================================\n\n")
#             f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Input Directory: {input_dir}\n")
#             f.write(f"File Pattern: {file_pattern}\n")
#             f.write(f"Files Processed: {len(mesh_files)}\n\n")
            
#             f.write("Results Summary:\n")
#             f.write("--------------\n\n")
            
#             for result in summary_results:
#                 if "error" in result:
#                     f.write(f"Mesh: {result['mesh']}\n")
#                     f.write(f"  ERROR: {result['error']}\n\n")
#                 else:
#                     f.write(f"Mesh: {result['mesh']}\n")
#                     f.write(f"  Vertices: {result['vertices']}\n")
#                     f.write(f"  Outlier count: {result['outlier_count']}\n")
#                     f.write(f"  Outlier percentage: {result['outlier_percentage']:.2f}%\n")
#                     f.write(f"  Concern level: {result['concern_level']}\n\n")
        
#         # Create comparison chart
#         if len(summary_results) > 1:
#             # Filter results to remove any with errors
#             valid_results = [r for r in summary_results if "error" not in r]
            
#             if valid_results:
#                 # Sort by outlier percentage
#                 valid_results.sort(key=lambda x: x["outlier_percentage"], reverse=True)
                
#                 # Create bar chart
#                 plt.figure(figsize=(12, 8))
                
#                 # Extract data
#                 mesh_names = [r["mesh"] for r in valid_results]
#                 percentages = [r["outlier_percentage"] for r in valid_results]
#                 colors = ['#ff7f7f' if r["concern_level"] == "HIGH" else
#                          '#ffbf7f' if r["concern_level"] == "MODERATE" else
#                          '#7fbfff' for r in valid_results]
                
#                 # Create bar chart
#                 bars = plt.bar(range(len(mesh_names)), percentages, color=colors)
                
#                 # Add labels and formatting
#                 plt.xticks(range(len(mesh_names)), mesh_names, rotation=90)
#                 plt.xlabel('Mesh File')
#                 plt.ylabel('Outlier Percentage (%)')
#                 plt.title('Outlier Percentage Comparison Across Meshes')
#                 plt.grid(axis='y', alpha=0.3)
                
#                 # Add threshold line
#                 plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='High Concern (5%)')
#                 plt.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Moderate Concern (1%)')
                
#                 plt.legend()
#                 plt.tight_layout()
                
#                 # Save chart
#                 plt.savefig(os.path.join(output_dir, f"outlier_comparison_{timestamp}.png"))
#                 plt.close()
    
#     print(f"\nBatch processing complete. Summary saved to {output_dir}")

# if __name__ == "__main__":

#     input_dir = "/envau/work/meca/users/dienye.h/B7_analysis/mesh/"
#     output_dir = "/envau/work/meca/users/dienye.h/mesh_analysis/statistical/"
#     # import argparse
    
#     # parser = argparse.ArgumentParser(description="Batch process multiple mesh files for statistical outlier detection")
#     # parser.add_argument("--/envau/work/meca/users/dienye.h/B7_analysis/mesh/", "-i", required=True, help="Directory containing mesh files")
#     # parser.add_argument("--/envau/work/meca/users/dienye.h/mesh_analysis/statistical/", "-o", default="outlier_analysis", help="Directory to save results")
#     # parser.add_argument("--pattern", "-p", default="*.gii", help="File pattern to match (default: *.gii)")
#     # parser.add_argument("--neighbors", "-k", type=int, default=30, help="Number of nearest neighbors to consider")
#     # parser.add_argument("--threshold", "-t", type=float, default=2.0, help="Outlier threshold in standard deviations")
    
#     # args = parser.parse_args()
    
#     batch_process_directory(
#         input_dir, 
#         output_dir
#     )

import numpy as np
import trimesh
import nibabel as nib
from sklearn.neighbors import NearestNeighbors
import time
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import scipy.sparse as sparse
from multiprocessing import Pool, cpu_count

def load_mesh(mesh_path):
    """
    Load a mesh file, supporting GIFTI (.gii) format.
    
    Parameters:
    -----------
    mesh_path : str
        Path to the mesh file
        
    Returns:
    --------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
    """
    print(f"Loading mesh from {mesh_path}...")
    
    # Check if file is a GIFTI file
    if mesh_path.lower().endswith('.gii'):
        return load_gifti(mesh_path)
    else:
        # Use trimesh for other file types
        mesh = trimesh.load_mesh(mesh_path)
        return mesh.vertices, mesh.faces

def load_gifti(gifti_path):
    """
    Load a GIFTI format mesh file.
    
    Parameters:
    -----------
    gifti_path : str
        Path to the GIFTI file
        
    Returns:
    --------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
    """
    try:
        # Load GIFTI file
        gifti_img = nib.load(gifti_path)
        
        # Extract vertices and faces from data arrays
        data_arrays = gifti_img.darrays
        
        vertices = None
        faces = None
        
        # Try to identify arrays by intent
        for da in data_arrays:
            if hasattr(da, 'intent') and da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                vertices = da.data
            elif hasattr(da, 'intent') and da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
                faces = da.data
        
        # If intent-based detection failed, try position-based
        if vertices is None and len(data_arrays) > 0:
            vertices = data_arrays[0].data
        if faces is None and len(data_arrays) > 1:
            faces = data_arrays[1].data
        
        # Ensure we have valid data
        if vertices is not None:
            vertices = vertices.astype(np.float64)
        else:
            raise ValueError("Could not extract vertices from GIFTI file")
            
        if faces is not None:
            faces = faces.astype(np.int32)
        else:
            raise ValueError("Could not extract faces from GIFTI file")
        
        return vertices, faces
        
    except Exception as e:
        raise ValueError(f"Error loading GIFTI file: {str(e)}")

def get_vertex_neighbors(vertices, faces):
    """
    Get neighboring vertices for each vertex.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
        
    Returns:
    --------
    neighbors : list
        List of neighboring vertices for each vertex
    """
    print("Computing vertex neighbors...")
    start_time = time.time()
    
    num_vertices = len(vertices)
    neighbors = [[] for _ in range(num_vertices)]
    
    # Build adjacency from faces
    for i, face in enumerate(faces):
        # For each vertex in the face, add the other two as neighbors
        neighbors[face[0]].extend([face[1], face[2]])
        neighbors[face[1]].extend([face[0], face[2]])
        neighbors[face[2]].extend([face[0], face[1]])
        
        # Print progress for large meshes
        if i % 100000 == 0 and i > 0:
            print(f"  Processing faces... {i}/{len(faces)}")
    
    # Remove duplicates (using sets is faster)
    for i in range(num_vertices):
        neighbors[i] = list(set(neighbors[i]))
    
    print(f"Neighbor computation completed in {time.time() - start_time:.2f} seconds")
    return neighbors

def smooth_mesh(vertices, neighbors, iterations=1, lambda_factor=0.5):
    """
    Apply Laplacian smoothing to a mesh.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    neighbors : list
        List of neighboring vertices for each vertex
    iterations : int
        Number of smoothing iterations
    lambda_factor : float
        Smoothing factor (0 to 1)
        
    Returns:
    --------
    smoothed_vertices : np.array
        Smoothed vertex coordinates
    """
    smoothed_vertices = vertices.copy()
    
    for _ in range(iterations):
        new_vertices = smoothed_vertices.copy()
        
        # Process each vertex
        for i in range(len(vertices)):
            if not neighbors[i]:
                continue
            
            # Calculate centroid of neighbors
            neighbor_centroid = np.mean(smoothed_vertices[neighbors[i]], axis=0)
            
            # Apply smoothing
            new_vertices[i] = smoothed_vertices[i] + lambda_factor * (neighbor_centroid - smoothed_vertices[i])
        
        smoothed_vertices = new_vertices
    
    return smoothed_vertices

def compute_statistical_outliers(vertices, k=30, std_ratio=2.0, batch_size=10000):
    """
    Identify statistical outliers based on distance to neighbors.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    k : int
        Number of nearest neighbors to consider
    std_ratio : float
        Number of standard deviations to use as threshold
    batch_size : int
        Size of batches for processing large meshes
        
    Returns:
    --------
    outlier_scores : np.array
        Outlier score for each vertex (higher = more likely to be an outlier)
    is_outlier : np.array
        Boolean array indicating if vertex is an outlier
    """
    start_time = time.time()
    num_vertices = len(vertices)
    
    print(f"Computing statistical outliers for {num_vertices} vertices (k={k}, std_ratio={std_ratio})...")
    
    # For large meshes, limit k to prevent excessive memory usage
    if num_vertices > 1000000 and k > 20:
        k = 20
        print(f"Reduced k to {k} due to large mesh size")
    
    # Initialize output array
    mean_distances = np.zeros(num_vertices)
    
    # Build KD-tree for efficient neighbor searches
    print("Building KD-tree for nearest neighbor search...")
    tree = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree', leaf_size=40, n_jobs=-1)
    tree.fit(vertices)
    
    # Process in batches to manage memory usage
    num_batches = int(np.ceil(num_vertices / batch_size))
    print(f"Processing {num_vertices} vertices in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_vertices)
        
        if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Processing batch {batch_idx+1}/{num_batches} (vertices {batch_start}-{batch_end})...")
        
        # Get vertices for this batch
        batch_vertices = vertices[batch_start:batch_end]
        
        # Find k nearest neighbors
        distances, _ = tree.kneighbors(batch_vertices)
        
        # Average distance to k nearest neighbors (excluding self)
        batch_mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Store results
        mean_distances[batch_start:batch_end] = batch_mean_distances
    
    # Compute global statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # Compute outlier scores
    outlier_scores = (mean_distances - global_mean) / global_std
    is_outlier = outlier_scores > std_ratio
    
    print(f"Statistical outlier computation completed in {time.time() - start_time:.2f} seconds")
    print(f"Identified {np.sum(is_outlier)} outliers out of {num_vertices} vertices ({np.mean(is_outlier)*100:.2f}%)")
    
    return outlier_scores, is_outlier

def compute_curvature(vertices, faces, neighbors):
    """
    Compute mean curvature at each vertex using discrete Laplacian.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
    neighbors : list
        List of neighboring vertices for each vertex
        
    Returns:
    --------
    curvature : np.array
        Mean curvature at each vertex
    """
    print("Computing curvature...")
    start_time = time.time()
    
    num_vertices = len(vertices)
    curvature = np.zeros(num_vertices)
    
    # Compute vertex normals
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    vertex_normals = mesh.vertex_normals
    
    # Process each vertex
    for i in range(num_vertices):
        if not neighbors[i]:
            continue
        
        # Get neighboring vertices
        neighbor_vertices = vertices[neighbors[i]]
        
        # Calculate centroid of neighbors
        centroid = np.mean(neighbor_vertices, axis=0)
        
        # Calculate Laplacian approximation (vertex - centroid)
        laplacian = vertices[i] - centroid
        
        # Project onto normal for mean curvature approximation
        curvature[i] = np.dot(laplacian, vertex_normals[i])
    
    print(f"Curvature computation completed in {time.time() - start_time:.2f} seconds")
    return curvature

def compute_multiscale_persistence(vertices, faces, neighbors, smooth_levels=[0, 5, 15, 30]):
    """
    Compute how features persist across multiple smoothing scales.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
    neighbors : list
        List of neighboring vertices for each vertex
    smooth_levels : list
        List of smoothing iterations to apply
        
    Returns:
    --------
    persistence_score : np.array
        Score indicating how much a vertex changes across scales (low = artifact)
    artifact_score : np.array
        Score indicating likelihood of being an artifact (high = likely artifact)
    """
    print("Computing multiscale persistence...")
    start_time = time.time()
    
    num_vertices = len(vertices)
    
    # Original mesh with no smoothing
    original_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Compute statistical outliers and curvature for the original mesh
    original_outliers, _ = compute_statistical_outliers(vertices, k=min(30, len(neighbors[0]) if neighbors[0] else 10))
    original_curvature = compute_curvature(vertices, faces, neighbors)
    
    # Normalize curvature
    original_curvature_norm = np.abs(original_curvature)
    if np.max(original_curvature_norm) > 0:
        original_curvature_norm = original_curvature_norm / np.max(original_curvature_norm)
    
    # Store original and smoothed values
    outlier_values = [original_outliers]
    curvature_values = [original_curvature_norm]
    
    # Compute smoothed versions
    for i, smooth_iter in enumerate(smooth_levels[1:], 1):
        print(f"Processing smoothing level {i}/{len(smooth_levels)-1} ({smooth_iter} iterations)...")
        
        # Smooth the mesh
        smoothed_vertices = smooth_mesh(vertices, neighbors, iterations=smooth_iter, lambda_factor=0.5)
        
        # Compute outliers and curvature for smoothed mesh
        smoothed_outliers, _ = compute_statistical_outliers(
            smoothed_vertices, 
            k=min(30, len(neighbors[0]) if neighbors[0] else 10)
        )
        
        smoothed_curvature = compute_curvature(smoothed_vertices, faces, neighbors)
        
        # Normalize curvature
        smoothed_curvature_norm = np.abs(smoothed_curvature)
        if np.max(smoothed_curvature_norm) > 0:
            smoothed_curvature_norm = smoothed_curvature_norm / np.max(smoothed_curvature_norm)
        
        # Store values
        outlier_values.append(smoothed_outliers)
        curvature_values.append(smoothed_curvature_norm)
    
    # Convert to numpy arrays for easier calculation
    outlier_values = np.array(outlier_values)
    curvature_values = np.array(curvature_values)
    
    # Compute persistence: how much values change across scales
    # High persistence = anatomical feature, Low persistence = artifact
    
    # Compute differences between original and smoothed versions
    outlier_diffs = np.zeros((len(smooth_levels)-1, num_vertices))
    curvature_diffs = np.zeros((len(smooth_levels)-1, num_vertices))
    
    for i in range(len(smooth_levels)-1):
        outlier_diffs[i] = np.abs(outlier_values[i+1] - outlier_values[0])
        curvature_diffs[i] = np.abs(curvature_values[i+1] - curvature_values[0])
    
    # Average differences across scales
    mean_outlier_diff = np.mean(outlier_diffs, axis=0)
    mean_curvature_diff = np.mean(curvature_diffs, axis=0)
    
    # Normalize differences
    if np.max(mean_outlier_diff) > 0:
        mean_outlier_diff = mean_outlier_diff / np.max(mean_outlier_diff)
    if np.max(mean_curvature_diff) > 0:
        mean_curvature_diff = mean_curvature_diff / np.max(mean_curvature_diff)
    
    # Combine metrics: high score = likely artifact
    # Features that disappear with smoothing and have high initial curvature/outlier values
    artifact_score = (mean_outlier_diff + mean_curvature_diff) / 2
    
    # Persistence score: high score = persistent feature (anatomical)
    persistence_score = 1.0 - artifact_score
    
    print(f"Multiscale persistence computation completed in {time.time() - start_time:.2f} seconds")
    return persistence_score, artifact_score

def visualize_mesh(vertices, faces, scores, output_path, colormap='viridis', title='Mesh Visualization'):
    """
    Export mesh with color-coded scores.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
    scores : np.array
        Score values for each vertex
    output_path : str
        Path to save the visualization
    colormap : str
        Matplotlib colormap to use
    title : str
        Title for the visualization
    """
    print(f"Visualizing mesh using colormap '{colormap}'...")
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Normalize scores to [0,1]
    if np.max(scores) > np.min(scores):
        normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        normalized_scores = np.zeros_like(scores)
    
    # Get colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    
    # Apply colormap
    colors = cmap(normalized_scores)[:, 0:3]
    
    # Create colored mesh
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
    
    # Export mesh
    colored_mesh.export(output_path)
    print(f"Visualization saved to {output_path}")

def analyze_mesh_quality(mesh_path, output_dir):
    """
    Analyze mesh quality using multiscale approach.
    
    Parameters:
    -----------
    mesh_path : str
        Path to the mesh file
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    result_summary : dict
        Dictionary with summary statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load mesh
    vertices, faces = load_mesh(mesh_path)
    num_vertices = len(vertices)
    
    print(f"Loaded mesh with {num_vertices} vertices and {len(faces)} faces")
    
    # Get vertex neighbors
    neighbors = get_vertex_neighbors(vertices, faces)
    
    # Compute artifact scores using multiscale approach
    smooth_levels = [0, 5, 15, 30]  # Different smoothing levels
    persistence_score, artifact_score = compute_multiscale_persistence(vertices, faces, neighbors, smooth_levels)
    
    # Save scores
    np.savetxt(os.path.join(output_dir, 'persistence_score.txt'), persistence_score)
    np.savetxt(os.path.join(output_dir, 'artifact_score.txt'), artifact_score)
    
    # Create visualizations
    visualize_mesh(
        vertices, faces, artifact_score,
        os.path.join(output_dir, 'artifact_visualization.ply'),
        colormap='viridis', title='Artifact Score'
    )
    
    visualize_mesh(
        vertices, faces, persistence_score,
        os.path.join(output_dir, 'persistence_visualization.ply'),
        colormap='viridis', title='Persistence Score'
    )
    
    # Create histograms
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plt.hist(artifact_score, bins=100)
    plt.title('Artifact Score Distribution')
    plt.xlabel('Artifact Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(122)
    plt.hist(persistence_score, bins=100)
    plt.title('Persistence Score Distribution')
    plt.xlabel('Persistence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    plt.close()
    
    # Calculate statistics
    high_artifact_threshold = np.percentile(artifact_score, 90)
    high_artifact_count = np.sum(artifact_score > high_artifact_threshold)
    high_artifact_percentage = (high_artifact_count / num_vertices) * 100
    
    # Determine quality level
    if high_artifact_percentage > 2.0:
        quality_level = "LOW"
    elif high_artifact_percentage > 0.5:
        quality_level = "MEDIUM"
    else:
        quality_level = "HIGH"
    
    # Write summary
    with open(os.path.join(output_dir, 'quality_summary.txt'), 'w') as f:
        f.write("Multiscale Mesh Quality Analysis Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Mesh: {os.path.basename(mesh_path)}\n")
        f.write(f"Vertices: {num_vertices}\n")
        f.write(f"Faces: {len(faces)}\n\n")
        f.write(f"Analysis Parameters:\n")
        f.write(f"- Smoothing levels: {smooth_levels}\n\n")
        f.write(f"Results:\n")
        f.write(f"- High artifact vertex count: {high_artifact_count}\n")
        f.write(f"- High artifact percentage: {high_artifact_percentage:.4f}%\n")
        f.write(f"- Mean artifact score: {np.mean(artifact_score):.4f}\n")
        f.write(f"- Max artifact score: {np.max(artifact_score):.4f}\n\n")
        f.write(f"Quality Assessment: {quality_level} QUALITY - ")
        
        if quality_level == "LOW":
            f.write("Significant artifacts detected\n")
        elif quality_level == "MEDIUM":
            f.write("Some artifacts present\n")
        else:
            f.write("Few artifacts detected\n")
        
        f.write("\nDetails:\n")
        f.write("The multiscale persistence approach distinguishes between true anatomical features and artifacts.\n")
        f.write("Features that persist across smoothing scales are likely anatomical, while features that quickly\n")
        f.write("disappear with smoothing are likely artifacts. The artifact score quantifies this distinction.\n")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    # Return summary
    return {
        "mesh": os.path.basename(mesh_path),
        "vertices": num_vertices,
        "faces": len(faces),
        "high_artifact_count": high_artifact_count,
        "high_artifact_percentage": high_artifact_percentage,
        "mean_artifact_score": np.mean(artifact_score),
        "max_artifact_score": np.max(artifact_score),
        "quality_level": quality_level
    }

def batch_process_directory(input_dir, output_dir, file_pattern="*.gii"):
    """
    Process all mesh files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing mesh files
    output_dir : str
        Directory to save results
    file_pattern : str
        Pattern to match files (e.g., "*.gii", "*.obj")
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    mesh_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not mesh_files:
        print(f"No files matching pattern '{file_pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(mesh_files)} files to process")
    
    # Create timestamp for this batch run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Summary results to collect
    summary_results = []
    
    # Process each file
    for i, mesh_path in enumerate(mesh_files):
        try:
            mesh_name = os.path.basename(mesh_path)
            mesh_output_dir = os.path.join(output_dir, os.path.splitext(mesh_name)[0])
            
            print(f"\n[{i+1}/{len(mesh_files)}] Processing {mesh_name}...")
            
            # Analyze this mesh
            result = analyze_mesh_quality(mesh_path, mesh_output_dir)
            
            # Add to summary
            summary_results.append(result)
            
        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            # Add error entry to summary
            summary_results.append({
                "mesh": os.path.basename(mesh_path),
                "error": str(e)
            })
    
    # Create summary report
    if summary_results:
        # CSV summary
        summary_df = pd.DataFrame(summary_results)
        summary_csv_path = os.path.join(output_dir, f"batch_summary_{timestamp}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        
        # Text summary
        with open(os.path.join(output_dir, f"batch_summary_{timestamp}.txt"), 'w') as f:
            f.write("Batch Multiscale Mesh Quality Analysis Summary\n")
            f.write("===========================================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {input_dir}\n")
            f.write(f"File Pattern: {file_pattern}\n")
            f.write(f"Files Processed: {len(mesh_files)}\n\n")
            
            f.write("Results Summary:\n")
            f.write("--------------\n\n")
            
            for result in summary_results:
                if "error" in result:
                    f.write(f"Mesh: {result['mesh']}\n")
                    f.write(f"  ERROR: {result['error']}\n\n")
                else:
                    f.write(f"Mesh: {result['mesh']}\n")
                    f.write(f"  Vertices: {result['vertices']}\n")
                    f.write(f"  High artifact count: {result['high_artifact_count']}\n")
                    f.write(f"  High artifact percentage: {result['high_artifact_percentage']:.4f}%\n")
                    f.write(f"  Quality level: {result['quality_level']}\n\n")
        
        # Create comparison chart
        if len(summary_results) > 1:
            # Filter results to remove any with errors
            valid_results = [r for r in summary_results if "error" not in r]
            
            if valid_results:
                # Sort by high_artifact_percentage
                valid_results.sort(key=lambda x: x["high_artifact_percentage"], reverse=True)
                
                # Create bar chart
                plt.figure(figsize=(12, 8))
                
                # Extract data
                mesh_names = [r["mesh"] for r in valid_results]
                percentages = [r["high_artifact_percentage"] for r in valid_results]
                colors = ['#ff7f7f' if r["quality_level"] == "LOW" else
                         '#ffbf7f' if r["quality_level"] == "MEDIUM" else
                         '#7fbfff' for r in valid_results]
                
                # Create bar chart
                bars = plt.bar(range(len(mesh_names)), percentages, color=colors)
                
                # Add labels and formatting
                plt.xticks(range(len(mesh_names)), mesh_names, rotation=90)
                plt.xlabel('Mesh File')
                plt.ylabel('High Artifact Percentage (%)')
                plt.title('Artifact Percentage Comparison Across Meshes')
                plt.grid(axis='y', alpha=0.3)
                
                # Add threshold lines
                plt.axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Low Quality (2.0%)')
                plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Quality (0.5%)')
                
                plt.legend()
                plt.tight_layout()
                
                # Save chart
                plt.savefig(os.path.join(output_dir, f"quality_comparison_{timestamp}.png"))
                plt.close()
    
    print(f"\nBatch processing complete. Summary saved to {output_dir}")

if __name__ == "__main__":
    
    input_dir = "/envau/work/meca/users/dienye.h/B7_analysis/mesh/"
    output_dir = "/envau/work/meca/users/dienye.h/mesh_analysis/statistical/"
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Multiscale mesh quality analysis for brain meshes")
    # parser.add_argument("--input_dir", "-i", required=True, help="Directory containing mesh files")
    # parser.add_argument("--output_dir", "-o", default="quality_analysis", help="Directory to save results")
    # parser.add_argument("--pattern", "-p", default="*.gii", help="File pattern to match (default: *.gii)")
    
    # args = parser.parse_args()
    
    batch_process_directory(
        input_dir, 
        output_dir, 
    )