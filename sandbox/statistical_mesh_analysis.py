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

def compute_statistical_outliers(vertices, k=30, std_ratio=2.0, batch_size=10000):
    """
    Identify statistical outliers based on distance to neighbors.
    Optimized for memory efficiency and speed.
    
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

def visualize_outliers(vertices, faces, outlier_scores, output_path, colormap='viridis'):
    """
    Export mesh with outlier visualization.
    
    Parameters:
    -----------
    vertices : np.array
        Vertex coordinates
    faces : np.array
        Face indices
    outlier_scores : np.array
        Outlier scores for each vertex
    output_path : str
        Path to save the visualization
    colormap : str
        Matplotlib colormap to use
    """
    print(f"Visualizing outliers using colormap '{colormap}'...")
    
    # Create temporary mesh for visualization
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Normalize scores to [0,1]
    if np.max(outlier_scores) > np.min(outlier_scores):
        normalized_scores = (outlier_scores - np.min(outlier_scores)) / (np.max(outlier_scores) - np.min(outlier_scores))
    else:
        normalized_scores = np.zeros_like(outlier_scores)
    
    # Get colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    
    # Apply colormap (red for high outlier scores)
    colors = cmap(normalized_scores)[:, 0:3]
    
    # Create colored mesh
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
    
    # Export mesh
    colored_mesh.export(output_path)
    print(f"Visualization saved to {output_path}")

def analyze_outliers(mesh_path, output_dir, k=30, std_ratio=2.0):
    """
    Analyze statistical outliers in a mesh and generate results.
    
    Parameters:
    -----------
    mesh_path : str
        Path to the mesh file
    output_dir : str
        Directory to save results
    k : int
        Number of nearest neighbors to consider
    std_ratio : float
        Number of standard deviations to use as threshold
        
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
    
    # Adjust parameters for very large meshes
    if num_vertices > 1000000:
        k = min(k, 20)
        batch_size = 5000
    elif num_vertices > 500000:
        k = min(k, 25)
        batch_size = 10000
    else:
        batch_size = 20000
    
    # Compute outliers
    outlier_scores, is_outlier = compute_statistical_outliers(
        vertices, k=k, std_ratio=std_ratio, batch_size=batch_size
    )
    
    # Save results
    np.savetxt(os.path.join(output_dir, 'outlier_scores.txt'), outlier_scores)
    np.savetxt(os.path.join(output_dir, 'is_outlier.txt'), is_outlier.astype(int))
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(outlier_scores, bins=100)
    plt.axvline(x=std_ratio, color='r', linestyle='--', label=f'Threshold ({std_ratio} std)')
    plt.title('Distribution of Outlier Scores')
    plt.xlabel('Outlier Score (standard deviations)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_histogram.png'))
    plt.close()
    
    # Visualize outliers
    visualize_outliers(
        vertices, faces, outlier_scores, 
        os.path.join(output_dir, 'outlier_visualization.ply')
    )
    
    # Calculate statistics
    outlier_count = np.sum(is_outlier)
    outlier_percentage = np.mean(is_outlier) * 100
    mean_score = np.mean(outlier_scores)
    max_score = np.max(outlier_scores)
    
    # Determine concern level
    if outlier_percentage > 5:
        concern_level = "HIGH"
    elif outlier_percentage > 1:
        concern_level = "MODERATE"
    else:
        concern_level = "LOW"
    
    # Write summary
    with open(os.path.join(output_dir, 'outlier_summary.txt'), 'w') as f:
        f.write("Statistical Outlier Analysis Summary\n")
        f.write("===================================\n\n")
        f.write(f"Mesh: {os.path.basename(mesh_path)}\n")
        f.write(f"Vertices: {num_vertices}\n")
        f.write(f"Faces: {len(faces)}\n\n")
        f.write(f"Analysis Parameters:\n")
        f.write(f"- Nearest neighbors (k): {k}\n")
        f.write(f"- Outlier threshold: {std_ratio} standard deviations\n\n")
        f.write(f"Results:\n")
        f.write(f"- Outlier count: {outlier_count}\n")
        f.write(f"- Outlier percentage: {outlier_percentage:.2f}%\n")
        f.write(f"- Mean outlier score: {mean_score:.4f}\n")
        f.write(f"- Maximum outlier score: {max_score:.4f}\n\n")
        f.write(f"Assessment: {concern_level} CONCERN - ")
        
        if concern_level == "HIGH":
            f.write("Large percentage of statistical outliers detected\n")
        elif concern_level == "MODERATE":
            f.write("Notable percentage of statistical outliers detected\n")
        else:
            f.write("Small percentage of statistical outliers detected\n")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    
    # Return summary for batch processing
    return {
        "mesh": os.path.basename(mesh_path),
        "vertices": num_vertices,
        "faces": len(faces),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage,
        "mean_score": mean_score,
        "max_score": max_score,
        "concern_level": concern_level
    }

def batch_process_directory(input_dir, output_dir, file_pattern="*.gii", k=30, std_ratio=2.0):
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
    k : int
        Number of nearest neighbors to consider
    std_ratio : float
        Number of standard deviations to use as threshold
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
            result = analyze_outliers(mesh_path, mesh_output_dir, k=k, std_ratio=std_ratio)
            
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
            f.write("Batch Statistical Outlier Analysis Summary\n")
            f.write("========================================\n\n")
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
                    f.write(f"  Outlier count: {result['outlier_count']}\n")
                    f.write(f"  Outlier percentage: {result['outlier_percentage']:.2f}%\n")
                    f.write(f"  Concern level: {result['concern_level']}\n\n")
        
        # Create comparison chart
        if len(summary_results) > 1:
            # Filter results to remove any with errors
            valid_results = [r for r in summary_results if "error" not in r]
            
            if valid_results:
                # Sort by outlier percentage
                valid_results.sort(key=lambda x: x["outlier_percentage"], reverse=True)
                
                # Create bar chart
                plt.figure(figsize=(12, 8))
                
                # Extract data
                mesh_names = [r["mesh"] for r in valid_results]
                percentages = [r["outlier_percentage"] for r in valid_results]
                colors = ['#ff7f7f' if r["concern_level"] == "HIGH" else
                         '#ffbf7f' if r["concern_level"] == "MODERATE" else
                         '#7fbfff' for r in valid_results]
                
                # Create bar chart
                bars = plt.bar(range(len(mesh_names)), percentages, color=colors)
                
                # Add labels and formatting
                plt.xticks(range(len(mesh_names)), mesh_names, rotation=90)
                plt.xlabel('Mesh File')
                plt.ylabel('Outlier Percentage (%)')
                plt.title('Outlier Percentage Comparison Across Meshes')
                plt.grid(axis='y', alpha=0.3)
                
                # Add threshold line
                plt.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='High Concern (5%)')
                plt.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Moderate Concern (1%)')
                
                plt.legend()
                plt.tight_layout()
                
                # Save chart
                plt.savefig(os.path.join(output_dir, f"outlier_comparison_{timestamp}.png"))
                plt.close()
    
    print(f"\nBatch processing complete. Summary saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process multiple mesh files for statistical outlier detection")
    parser.add_argument("--/envau/work/meca/users/dienye.h/B7_analysis/mesh/", "-i", required=True, help="Directory containing mesh files")
    parser.add_argument("--/envau/work/meca/users/dienye.h/mesh_analysis/statistical/", "-o", default="outlier_analysis", help="Directory to save results")
    parser.add_argument("--pattern", "-p", default="*.gii", help="File pattern to match (default: *.gii)")
    parser.add_argument("--neighbors", "-k", type=int, default=30, help="Number of nearest neighbors to consider")
    parser.add_argument("--threshold", "-t", type=float, default=2.0, help="Outlier threshold in standard deviations")
    
    args = parser.parse_args()
    
    batch_process_directory(
        args.input_dir, 
        args.output_dir, 
        file_pattern=args.pattern,
        k=args.neighbors, 
        std_ratio=args.threshold
    )