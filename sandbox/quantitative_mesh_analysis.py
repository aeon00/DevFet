import numpy as np
import trimesh
import scipy.sparse as sparse
import nibabel as nib
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
import glob
import argparse
import numpy as np
import pandas as pd

class CorticalMeshQuality:
    """
    A class for analyzing the quality of cortical surface meshes, focusing on
    distinguishing between natural cortical folding and segmentation artifacts.
    """
    
    def __init__(self, mesh_path):
        """
        Initialize with a mesh file.
        
        Parameters:
        -----------
        mesh_path : str
            Path to the mesh file (.obj, .stl, .gii, etc.)
        """
        # Check if file is a GIFTI file
        if mesh_path.lower().endswith('.gii'):
            self._load_gifti(mesh_path)
        else:
            # Use trimesh for other file types
            self.mesh = trimesh.load_mesh(mesh_path)
            self.vertices = self.mesh.vertices
            self.faces = self.mesh.faces
            
    def _load_gifti(self, gifti_path):
        """
        Load a GIFTI format mesh file.
        
        Parameters:
        -----------
        gifti_path : str
            Path to the GIFTI file
        """
        try:
            # Load GIFTI file
            gifti_img = nib.load(gifti_path)
            
            # Extract vertices and faces
            data_arrays = gifti_img.darrays
            
            # GIFTI files typically have vertices as the first array and faces as the second,
            # but this can vary. Let's try to identify them by their intent
            vertices = None
            faces = None
            
            for da in data_arrays:
                # Check intent to determine content type
                if da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
                    vertices = da.data
                elif da.intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
                    faces = da.data
            
            # If intent-based detection failed, try position-based (common convention)
            if vertices is None and len(data_arrays) > 0:
                vertices = data_arrays[0].data
            if faces is None and len(data_arrays) > 1:
                faces = data_arrays[1].data
            
            # Convert to appropriate data types
            if vertices is not None:
                self.vertices = vertices.astype(np.float64)
            else:
                raise ValueError("Could not extract vertices from GIFTI file")
                
            if faces is not None:
                self.faces = faces.astype(np.int32)
            else:
                raise ValueError("Could not extract faces from GIFTI file")
            
            # Create a trimesh object for compatibility with the rest of the code
            self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            self.num_vertices = len(self.vertices)
            
            # Compute basic mesh properties
            self.vertex_normals = self.mesh.vertex_normals
            self.face_normals = self.mesh.face_normals
            self.vertex_neighbors = self._get_vertex_neighbors()
            
            # Pre-compute Laplacian for spectral analysis
            self.laplacian = self._compute_laplacian()
        except:
            return None
        
    def _get_vertex_neighbors(self):
        """Get neighboring vertices for each vertex."""
        neighbors = [[] for _ in range(self.num_vertices)]
        
        # Find adjacent vertices through faces
        for face in self.faces:
            for i in range(3):
                neighbors[face[i]].extend([face[(i+1)%3], face[(i+2)%3]])
        
        # Remove duplicates
        for i in range(self.num_vertices):
            neighbors[i] = list(set(neighbors[i]))
            
        return neighbors
    
    def _compute_laplacian(self):
        """Compute the Laplace-Beltrami operator (cotangent weights)."""
        vertices = self.vertices
        faces = self.faces
        
        # Initialize sparse matrix entries
        i_idx = []
        j_idx = []
        values = []
        
        # For each face, compute cotangent weights
        for face in faces:
            # Get vertices of the face
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            
            # Compute edges
            e0 = v2 - v1  # edge opposite to vertex 0
            e1 = v0 - v2  # edge opposite to vertex 1
            e2 = v1 - v0  # edge opposite to vertex 2
            
            # Normalize edges
            e0_norm = np.linalg.norm(e0)
            e1_norm = np.linalg.norm(e1)
            e2_norm = np.linalg.norm(e2)
            
            if e0_norm * e1_norm * e2_norm == 0:
                continue  # Skip degenerate triangles
                
            e0 = e0 / e0_norm
            e1 = e1 / e1_norm
            e2 = e2 / e2_norm
            
            # Compute cotangents using dot product
            # cot(angle) = cos(angle) / sin(angle) = dot(e1, e2) / |cross(e1, e2)|
            cot0 = np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))
            cot1 = np.dot(e2, e0) / np.linalg.norm(np.cross(e2, e0))
            cot2 = np.dot(e0, e1) / np.linalg.norm(np.cross(e0, e1))
            
            # Handle numerical instabilities
            cot0 = max(min(cot0, 100), -100)
            cot1 = max(min(cot1, 100), -100)
            cot2 = max(min(cot2, 100), -100)
            
            # Add cotangent weights to sparse matrix entries
            i_idx.extend([face[0], face[1], face[2], face[1], face[0], face[2], face[0], face[2], face[1]])
            j_idx.extend([face[0], face[1], face[2], face[0], face[1], face[1], face[2], face[0], face[2]])
            values.extend([0, 0, 0, cot2/2, cot2/2, cot0/2, cot0/2, cot1/2, cot1/2])
        
        # Create sparse matrix
        L = sparse.csr_matrix((values, (i_idx, j_idx)), shape=(self.num_vertices, self.num_vertices))
        
        # Make Laplacian symmetric by averaging with its transpose
        L = (L + L.T) / 2
        
        # Ensure row sum is zero (important for Laplacian properties)
        row_sum = L.sum(axis=1).A.flatten()
        for i in range(self.num_vertices):
            L[i, i] = -row_sum[i]
        
        return L
    
    def compute_curvature(self):
        """
        Compute mean curvature at each vertex using the Laplacian operator.
        
        Returns:
        --------
        mean_curvature : np.array
            Mean curvature at each vertex
        """
        # Apply Laplacian to vertex positions
        H = np.zeros(self.num_vertices)
        for dim in range(3):
            x = self.vertices[:, dim]
            Lx = self.laplacian.dot(x)
            H += Lx**2
        
        H = np.sqrt(H) / 2
        
        return H
    
    def compute_local_variation(self, k=10):
        """
        Compute local surface variation (measure of non-planarity).
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider
            
        Returns:
        --------
        variation : np.array
            Surface variation at each vertex (0 = planar, 1 = isotropic)
        """
        # Find k nearest neighbors for each vertex
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(self.vertices)
        _, indices = nbrs.kneighbors(self.vertices)
        
        variations = np.zeros(self.num_vertices)
        
        for i in range(self.num_vertices):
            # Get neighbors for vertex i (excluding self)
            neighbors = indices[i, 1:]
            neighbor_points = self.vertices[neighbors]
            
            # Compute local covariance matrix
            centroid = np.mean(neighbor_points, axis=0)
            centered = neighbor_points - centroid
            cov = np.dot(centered.T, centered) / k
            
            # Compute eigenvalues
            if np.all(cov == 0):
                variations[i] = 0
                continue
                
            evals = np.linalg.eigvalsh(cov)
            evals = np.sort(evals)
            
            # Surface variation = smallest eigenvalue / sum of eigenvalues
            if np.sum(evals) > 0:
                variations[i] = evals[0] / np.sum(evals)
        
        return variations
    
    def compute_statistical_outliers(self, k=30, std_ratio=2.0):
        """
        Identify statistical outliers based on distance to neighbors.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors to consider
        std_ratio : float
            Number of standard deviations to use as threshold
            
        Returns:
        --------
        outlier_scores : np.array
            Outlier score for each vertex (higher = more likely to be an outlier)
        is_outlier : np.array
            Boolean array indicating if vertex is an outlier
        """
        # Find k nearest neighbors for each vertex
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(self.vertices)
        distances, _ = nbrs.kneighbors(self.vertices)
        
        # Average distance to k nearest neighbors (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Compute outlier score based on local neighborhood statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        outlier_scores = (mean_distances - global_mean) / global_std
        is_outlier = outlier_scores > std_ratio
        
        return outlier_scores, is_outlier
    
    def compute_spectral_components(self, num_components=50):
        """
        Compute spectral components of the mesh using Laplacian eigenfunctions.
        
        Parameters:
        -----------
        num_components : int
            Number of spectral components to compute
            
        Returns:
        --------
        eigenvalues : np.array
            Eigenvalues of the Laplacian
        eigenfunctions : np.array
            Eigenfunctions of the Laplacian
        """
        # Compute the smallest eigenvalues and eigenvectors of the Laplacian
        eigenvalues, eigenfunctions = eigsh(self.laplacian, k=num_components, 
                                          which='SM', sigma=0)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenfunctions = eigenfunctions[:, idx]
        
        return eigenvalues, eigenfunctions
    
    def analyze_frequency_distribution(self, num_components=100):
        """
        Analyze the frequency distribution of the mesh.
        
        Parameters:
        -----------
        num_components : int
            Number of spectral components to compute
            
        Returns:
        --------
        high_freq_content : np.array
            Amount of high-frequency content at each vertex
        """
        # Compute spectral components
        eigenvalues, eigenfunctions = self.compute_spectral_components(num_components)
        
        # Define high-frequency components (e.g., top 70%)
        cutoff_idx = int(num_components * 0.3)  # 30% lowest frequencies are considered "low"
        
        # Compute contribution of high-frequency components at each vertex
        high_freq_content = np.zeros(self.num_vertices)
        for i in range(cutoff_idx, num_components):
            # Weight by eigenvalue magnitude (higher eigenvalue = higher frequency)
            contribution = eigenfunctions[:, i]**2 * eigenvalues[i]
            high_freq_content += contribution
        
        # Normalize
        if np.max(high_freq_content) > 0:
            high_freq_content = high_freq_content / np.max(high_freq_content)
        
        return high_freq_content
    
    def compute_scale_persistence(self, scales=[0, 0.5, 1.0, 2.0, 4.0]):
        """
        Analyze how features persist across different smoothing scales.
        
        Parameters:
        -----------
        scales : list
            Smoothing scales to analyze
            
        Returns:
        --------
        persistence : np.array
            Persistence score for each vertex (higher = more persistent across scales)
        """
        # Store mean curvature at each scale
        curvatures = []
        
        # Compute original curvature
        original_mesh = self.mesh.copy()
        self.mesh = original_mesh
        curvatures.append(self.compute_curvature())
        
        # Compute curvature at each smoothing scale
        for scale in scales[1:]:
            # Create smoothed mesh
            smoothed = original_mesh.copy()
            
            # Apply Laplacian smoothing
            if scale > 0:
                for _ in range(int(scale * 10)):  # Convert scale to iteration count
                    new_vertices = np.zeros_like(smoothed.vertices)
                    for i in range(len(smoothed.vertices)):
                        # Get all adjacent vertices
                        adj_vertices = [smoothed.vertices[j] for j in self.vertex_neighbors[i]]
                        if adj_vertices:
                            # Simple average of neighboring vertices
                            new_vertices[i] = np.mean(adj_vertices, axis=0)
                        else:
                            new_vertices[i] = smoothed.vertices[i]
                    
                    # Update vertices with slight damping to prevent instability
                    smoothed.vertices = 0.9 * new_vertices + 0.1 * smoothed.vertices
            
            # Set the mesh to the smoothed version and compute curvature
            self.mesh = smoothed
            self.vertices = smoothed.vertices
            curvatures.append(self.compute_curvature())
        
        # Restore original mesh
        self.mesh = original_mesh
        self.vertices = original_mesh.vertices
        
        # Compute persistence score
        curvatures = np.array(curvatures)
        
        # Normalize curvatures across scales
        for i in range(len(scales)):
            if np.max(curvatures[i]) > 0:
                curvatures[i] = curvatures[i] / np.max(curvatures[i])
        
        # Compute variance across scales for each vertex
        persistence = np.std(curvatures, axis=0)
        
        # Invert so that high values = high persistence
        persistence = 1 - (persistence / np.max(persistence) if np.max(persistence) > 0 else persistence)
        
        return persistence
    
    def compute_quality_metric(self):
        """
        Compute a combined quality metric for the cortical mesh.
        
        Returns:
        --------
        quality : np.array
            Quality score for each vertex (lower = potential segmentation issues)
        """
        # Compute individual metrics
        print("Computing curvature...")
        curvature = self.compute_curvature()
        
        print("Computing local variation...")
        variation = self.compute_local_variation()
        
        print("Computing statistical outliers...")
        outlier_scores, _ = self.compute_statistical_outliers()
        
        print("Computing frequency distribution...")
        high_freq = self.analyze_frequency_distribution()
        
        print("Computing scale persistence...")
        persistence = self.compute_scale_persistence()
        
        # Normalize each metric to [0, 1]
        curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature)) if np.max(curvature) > np.min(curvature) else np.zeros_like(curvature)
        
        # Compute combined quality metric
        # Low quality = high curvature + high variation + high outlier score + high frequency + low persistence
        quality = 1.0 - (0.2 * curvature + 0.2 * variation + 0.3 * (outlier_scores / np.max(np.abs(outlier_scores)) if np.max(np.abs(outlier_scores)) > 0 else outlier_scores) + 0.2 * high_freq - 0.3 * persistence)
        
        return quality
    
    def visualize_quality(self, quality, output_path=None, colormap='viridis'):
        """
        Visualize the quality metric on the mesh.
        
        Parameters:
        -----------
        quality : np.array
            Quality metric for each vertex
        output_path : str, optional
            Path to save the visualization
        colormap : str
            Matplotlib colormap to use
        """
        # Convert to trimesh colors (nx3 array of RGB values)
        import matplotlib.cm as cm
        
        # Normalize quality to [0, 1]
        if np.max(quality) > np.min(quality):
            normalized_quality = (quality - np.min(quality)) / (np.max(quality) - np.min(quality))
        else:
            normalized_quality = np.zeros_like(quality)
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Apply colormap to create RGB colors
        colors = cmap(normalized_quality)[:, 0:3]
        
        # Create a colored mesh
        colored_mesh = self.mesh.copy()
        colored_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
        
        # Save or display the mesh
        if output_path:
            colored_mesh.export(output_path)
            print(f"Visualization saved to {output_path}")
        
        return colored_mesh

    def save_quality_values(self, quality, output_path):
        """
        Save quality values to a file.
        
        Parameters:
        -----------
        quality : np.array
            Quality values for each vertex
        output_path : str
            Path to save the quality values
        """
        np.savetxt(output_path, quality)
        print(f"Quality values saved to {output_path}")
    
    def generate_quality_report(self, output_dir):
        """
        Generate a comprehensive quality report.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the report
        
        Returns:
        --------
        quality : np.array
            Overall quality metric
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute all metrics
        print("Computing all metrics...")
        curvature = self.compute_curvature()
        variation = self.compute_local_variation()
        outlier_scores, is_outlier = self.compute_statistical_outliers()
        high_freq = self.analyze_frequency_distribution()
        persistence = self.compute_scale_persistence()
        
        # Compute overall quality
        quality = self.compute_quality_metric()
        
        # Save quality mesh
        quality_mesh = self.visualize_quality(quality)
        quality_mesh.export(os.path.join(output_dir, 'quality_visualization.ply'))
        
        # Save individual metric values
        np.savetxt(os.path.join(output_dir, 'curvature.txt'), curvature)
        np.savetxt(os.path.join(output_dir, 'variation.txt'), variation)
        np.savetxt(os.path.join(output_dir, 'outlier_scores.txt'), outlier_scores)
        np.savetxt(os.path.join(output_dir, 'high_frequency.txt'), high_freq)
        np.savetxt(os.path.join(output_dir, 'persistence.txt'), persistence)
        np.savetxt(os.path.join(output_dir, 'quality.txt'), quality)
        
        # Create histogram plots
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.hist(curvature, bins=50)
        plt.title('Curvature Distribution')
        
        plt.subplot(232)
        plt.hist(variation, bins=50)
        plt.title('Local Variation Distribution')
        
        plt.subplot(233)
        plt.hist(outlier_scores, bins=50)
        plt.title('Outlier Score Distribution')
        
        plt.subplot(234)
        plt.hist(high_freq, bins=50)
        plt.title('High Frequency Content Distribution')
        
        plt.subplot(235)
        plt.hist(persistence, bins=50)
        plt.title('Scale Persistence Distribution')
        
        plt.subplot(236)
        plt.hist(quality, bins=50)
        plt.title('Overall Quality Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metric_distributions.png'))
        
        # Calculate summary statistics
        outlier_percentage = np.sum(is_outlier) / len(is_outlier) * 100
        low_quality_threshold = np.percentile(quality, 10)  # Bottom 10% considered low quality
        low_quality_percentage = np.sum(quality < low_quality_threshold) / len(quality) * 100
        
        # Write summary report
        with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
            f.write("Cortical Mesh Quality Analysis Report\n")
            f.write("====================================\n\n")
            
            f.write(f"Number of vertices: {self.num_vertices}\n")
            f.write(f"Number of faces: {len(self.faces)}\n\n")
            
            f.write("Summary Statistics:\n")
            f.write(f"- Mean curvature: {np.mean(curvature):.6f}\n")
            f.write(f"- Max curvature: {np.max(curvature):.6f}\n")
            f.write(f"- Mean local variation: {np.mean(variation):.6f}\n")
            f.write(f"- Percentage of statistical outliers: {outlier_percentage:.2f}%\n")
            f.write(f"- Percentage of low quality vertices: {low_quality_percentage:.2f}%\n\n")
            
            f.write("Quality Assessment:\n")
            if low_quality_percentage > 20:
                f.write("- HIGH CONCERN: Large percentage of low quality vertices detected\n")
            elif low_quality_percentage > 10:
                f.write("- MODERATE CONCERN: Notable percentage of low quality vertices detected\n")
            else:
                f.write("- LOW CONCERN: Small percentage of low quality vertices detected\n")
                
            if outlier_percentage > 5:
                f.write("- HIGH CONCERN: Large percentage of statistical outliers detected\n")
            elif outlier_percentage > 1:
                f.write("- MODERATE CONCERN: Notable percentage of statistical outliers detected\n")
            else:
                f.write("- LOW CONCERN: Small percentage of statistical outliers detected\n")
        
        print(f"Quality report generated in {output_dir}")
        return quality


def analyze_directory(input_dir, output_dir, file_pattern="*.gii"):
    """
    Analyze all GIFTI files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing GIFTI files
    output_dir : str
        Directory to save analysis results
    file_pattern : str
        Pattern to match files (default: "*.gii")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not files:
        print(f"No files matching pattern '{file_pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to analyze")
    
    # Results summary
    summary_data = []
    
    # Process each file
    for i, file_path in enumerate(files):
        try:
            filename = os.path.basename(file_path)
            print(f"\n[{i+1}/{len(files)}] Analyzing {filename}...")
            
            # Create subdirectory for this file
            file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Analyze mesh
            analyzer = CorticalMeshQuality(file_path)
            
            # Generate quality report
            quality = analyzer.generate_quality_report(file_output_dir)
            
            # Extract summary statistics
            outlier_scores, is_outlier = analyzer.compute_statistical_outliers()
            outlier_percentage = np.sum(is_outlier) / len(is_outlier) * 100
            
            low_quality_threshold = np.percentile(quality, 10)  # Bottom 10% considered low quality
            low_quality_percentage = np.sum(quality < low_quality_threshold) / len(quality) * 100
            
            # Add to summary
            summary_data.append({
                'Filename': filename,
                'Vertex Count': len(analyzer.vertices),
                'Face Count': len(analyzer.faces),
                'Mean Quality': np.mean(quality),
                'Min Quality': np.min(quality),
                'Outlier Percentage': outlier_percentage,
                'Low Quality Percentage': low_quality_percentage
            })
            
            print(f"Analysis of {filename} complete")
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Create summary report
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "analysis_summary.csv"), index=False)
        
        # Also save as text file
        with open(os.path.join(output_dir, "analysis_summary.txt"), 'w') as f:
            f.write("Cortical Mesh Quality Analysis Summary\n")
            f.write("=====================================\n\n")
            
            for i, row in summary_df.iterrows():
                f.write(f"File: {row['Filename']}\n")
                f.write(f"  Vertices: {row['Vertex Count']}\n")
                f.write(f"  Faces: {row['Face Count']}\n")
                f.write(f"  Mean Quality: {row['Mean Quality']:.4f}\n")
                f.write(f"  Min Quality: {row['Min Quality']:.4f}\n")
                f.write(f"  Outlier Percentage: {row['Outlier Percentage']:.2f}%\n")
                f.write(f"  Low Quality Percentage: {row['Low Quality Percentage']:.2f}%\n")
                
                # Add judgment
                if row['Low Quality Percentage'] > 20:
                    f.write("  Assessment: HIGH CONCERN - Large percentage of low quality vertices\n")
                elif row['Low Quality Percentage'] > 10:
                    f.write("  Assessment: MODERATE CONCERN - Notable percentage of low quality vertices\n")
                else:
                    f.write("  Assessment: LOW CONCERN - Small percentage of low quality vertices\n")
                
                f.write("\n")
        
        print(f"\nAnalysis summary saved to {os.path.join(output_dir, 'analysis_summary.csv')}")

if __name__ == "__main__":
    input_dir = '/envau/work/meca/users/dienye.h/B7_analysis/mesh/'
    output_dir = '/envau/work/meca/users/dienye.h/mesh_analysis/'
    
    analyze_directory(input_dir, output_dir)