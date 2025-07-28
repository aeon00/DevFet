# import slam.io as sio
# import slam.texture as stex
# import slam.curvature as scurv
# import os
# from slam.differential_geometry import laplacian_mesh_smoothing

# def ensure_dir_exists(directory):
#     """Create directory if it doesn't exist"""
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# mesh_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/missing_mean_curv_mesh'
# mean_tex_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/mean_curv_tex/'
# principal_tex_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/principal_curv_tex/'

# # Ensure output directories exist
# ensure_dir_exists(mean_tex_dir)
# ensure_dir_exists(principal_tex_dir)

# for mesh_file in os.listdir(mesh_dir):
#     if not mesh_file.endswith(('.gii', '.mesh', '.ply')):  # Add appropriate mesh extensions
#         continue
    
#     # Extract filename without extension for texture naming
#     filename = os.path.splitext(mesh_file)[0]
    
#     # Define the path where the mean curvature texture would be saved
#     mean_tex_path = os.path.join(mean_tex_dir, 'filt_mean_curv_{}.gii'.format(filename))
    
#     # Check if mean curvature texture already exists
#     if os.path.exists(mean_tex_path):
#         print(f"Mean curvature texture already exists for {mesh_file}, skipping...")
#         continue
    
#     print(f"Processing mesh: {mesh_file}")
    
#     # Load and process the mesh
#     mesh_path = os.path.join(mesh_dir, mesh_file)
#     mesh = sio.load_mesh(mesh_path)
    
#     # CURVATURE
#     print("compute the mean curvature")
#     PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
#         scurv.curvatures_and_derivatives(mesh)
#     tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
    
#     # Save principal curvature texture
#     principal_tex_path = os.path.join(principal_tex_dir, 'principal_curv_{}.gii'.format(filename))
#     sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
    
#     # Compute and save mean curvature texture
#     mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
#     tex_mean_curv = stex.TextureND(mean_curv)
#     tex_mean_curv.z_score_filtering(z_thresh=3)
    
#     sio.write_texture(tex_mean_curv, mean_tex_path)
#     print(f"Completed processing: {mesh_file}")

# print("All meshes processed!")

# import slam.io as sio
# import slam.texture as stex
# import slam.curvature as scurv
# import slam.topology as stop
# import slam.remeshing as srem
# import os
# import numpy as np
# from slam.differential_geometry import laplacian_mesh_smoothing

# def clean_mesh(mesh, min_area_threshold=1e-10):
#     """Clean mesh by removing degenerate triangles and isolated vertices"""
    
#     # Remove degenerate triangles (very small area)
#     face_areas = []
#     valid_faces = []
    
#     for i, face in enumerate(mesh.faces):
#         v0, v1, v2 = mesh.vertices[face]
#         # Calculate triangle area using cross product
#         area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
#         if area > min_area_threshold:
#             face_areas.append(area)
#             valid_faces.append(face)
    
#     if len(valid_faces) < len(mesh.faces):
#         print(f"Removed {len(mesh.faces) - len(valid_faces)} degenerate triangles")
#         # Create new mesh with valid faces only
#         mesh.faces = np.array(valid_faces)
    
#     # Remove isolated vertices
#     mesh = stop.remove_isolated_vertices(mesh)
    
#     return mesh

# def robust_curvature_computation(mesh, max_attempts=3):
#     """Compute curvature with multiple fallback strategies"""
    
#     original_mesh = mesh.copy()
    
#     for attempt in range(max_attempts):
#         try:
#             print(f"Curvature computation attempt {attempt + 1}")
            
#             if attempt == 0:
#                 # First attempt: original mesh
#                 current_mesh = mesh
#             elif attempt == 1:
#                 # Second attempt: more aggressive smoothing
#                 print("Applying additional smoothing...")
#                 current_mesh = laplacian_mesh_smoothing(original_mesh, nb_iter=10, dt=0.05)
#             else:
#                 # Third attempt: remesh to improve quality
#                 print("Attempting remeshing...")
#                 try:
#                     current_mesh = srem.isotropic_remeshing(original_mesh)
#                 except:
#                     # If remeshing fails, use heavily smoothed version
#                     current_mesh = laplacian_mesh_smoothing(original_mesh, nb_iter=20, dt=0.01)
            
#             # Clean the mesh
#             current_mesh = clean_mesh(current_mesh)
            
#             # Validate mesh quality
#             if not validate_mesh_quality(current_mesh):
#                 continue
                
#             # Attempt curvature computation
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("ignore")
#                 PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
#                     scurv.curvatures_and_derivatives(current_mesh)
            
#             # Check if computation was successful
#             if (PrincipalCurvatures is not None and 
#                 not np.all(np.isnan(PrincipalCurvatures)) and 
#                 not np.all(np.isinf(PrincipalCurvatures))):
                
#                 print(f"Curvature computation successful on attempt {attempt + 1}")
#                 return PrincipalCurvatures, PrincipalDir1, PrincipalDir2, current_mesh
                
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed: {str(e)}")
#             continue
    
#     print("All curvature computation attempts failed")
#     return None, None, None, None

# def validate_mesh_quality(mesh):
#     """Validate mesh quality before curvature computation"""
    
#     # Check for minimum number of vertices and faces
#     if mesh.vertices.shape[0] < 10 or mesh.faces.shape[0] < 10:
#         print("Mesh too small for reliable curvature computation")
#         return False
    
#     # Check for valid coordinates
#     if (np.any(np.isnan(mesh.vertices)) or np.any(np.isinf(mesh.vertices)) or
#         np.any(np.isnan(mesh.faces)) or np.any(np.isinf(mesh.faces))):
#         print("Invalid coordinates detected")
#         return False
    
#     # Check triangle quality (aspect ratio)
#     min_quality = check_triangle_quality(mesh)
#     if min_quality < 0.01:  # Very poor quality triangles
#         print(f"Poor triangle quality detected: {min_quality}")
#         return False
    
#     return True

# def check_triangle_quality(mesh):
#     """Check triangle quality (aspect ratio)"""
#     qualities = []
    
#     for face in mesh.faces:
#         v0, v1, v2 = mesh.vertices[face]
        
#         # Calculate edge lengths
#         edge1 = np.linalg.norm(v1 - v0)
#         edge2 = np.linalg.norm(v2 - v1)
#         edge3 = np.linalg.norm(v0 - v2)
        
#         # Calculate area
#         area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        
#         # Calculate quality (area / perimeter^2)
#         if area > 0:
#             perimeter = edge1 + edge2 + edge3
#             quality = 4 * np.sqrt(3) * area / (perimeter ** 2)
#             qualities.append(quality)
    
#     return min(qualities) if qualities else 0

# def ensure_dir_exists(directory):
#     """Create directory if it doesn't exist"""
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # Main processing code with robust curvature computation
# import warnings

# mesh_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/missing_mean_curv_mesh'
# mean_tex_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/mean_curv_tex/'
# principal_tex_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/principal_curv_tex/'

# # Ensure output directories exist
# ensure_dir_exists(mean_tex_dir)
# ensure_dir_exists(principal_tex_dir)

# for mesh_file in os.listdir(mesh_dir):
#     if not mesh_file.endswith(('.gii', '.mesh', '.ply')):
#         continue
    
#     filename = os.path.splitext(mesh_file)[0]
#     mean_tex_path = os.path.join(mean_tex_dir, 'filt_mean_curv_{}.gii'.format(filename))
    
#     if os.path.exists(mean_tex_path):
#         print(f"Mean curvature texture already exists for {mesh_file}, skipping...")
#         continue
    
#     print(f"Processing mesh: {mesh_file}")
    
#     try:
#         # Load mesh
#         mesh_path = os.path.join(mesh_dir, mesh_file)
#         mesh = sio.load_mesh(mesh_path)
        
#         # Initial mesh cleaning and smoothing
#         mesh = clean_mesh(mesh)
#         mesh = laplacian_mesh_smoothing(mesh, nb_iter=5, dt=0.1)
        
#         # Robust curvature computation
#         PrincipalCurvatures, PrincipalDir1, PrincipalDir2, processed_mesh = \
#             robust_curvature_computation(mesh)
        
#         if PrincipalCurvatures is None:
#             print(f"Failed to compute curvatures for {mesh_file}, skipping...")
#             continue
        
#         # Process and save results
#         tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
#         principal_tex_path = os.path.join(principal_tex_dir, 'principal_curv_{}.gii'.format(filename))
#         sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
        
#         # Compute mean curvature
#         mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        
#         # Handle NaN values
#         if np.any(np.isnan(mean_curv)):
#             nan_ratio = np.sum(np.isnan(mean_curv)) / len(mean_curv)
#             if nan_ratio > 0.1:
#                 print(f"Too many NaN values ({nan_ratio:.2%}) in {mesh_file}, skipping...")
#                 continue
#             mean_curv = np.nan_to_num(mean_curv, nan=0.0)
        
#         tex_mean_curv = stex.TextureND(mean_curv)
#         tex_mean_curv.z_score_filtering(z_thresh=3)
#         sio.write_texture(tex_mean_curv, mean_tex_path)
        
#         print(f"Successfully processed: {mesh_file}")
        
#     except Exception as e:
#         print(f"Error processing {mesh_file}: {str(e)}")
#         continue

# print("Processing complete!")

#!/usr/bin/env python3
"""
Script to check if all generated meshes have their corresponding mean curvature files.

This script verifies the correspondence between:
1. Smoothed meshes in /scratch/hdienye/dhcp_full_info/mesh/
2. Mean curvature textures in /scratch/hdienye/dhcp_full_info/mean_curv_tex/

Based on the processing pipeline in the provided code.
"""

import os
import glob
import pandas as pd
from pathlib import Path

def extract_base_filename(filepath, prefix_to_remove=""):
    """
    Extract the base filename without directory and specific prefixes.
    
    Parameters:
    filepath: Path to the file
    prefix_to_remove: Prefix to remove from filename (e.g., 'smooth_5_', 'filt_mean_curv_')
    
    Returns:
    str: Base filename
    """
    filename = os.path.basename(filepath)
    if prefix_to_remove and filename.startswith(prefix_to_remove):
        filename = filename[len(prefix_to_remove):]
    return filename

def check_mesh_curvature_correspondence():
    """
    Check correspondence between generated meshes and mean curvature files.
    """
    # Define paths based on the original code
    mesh_path = "/scratch/hdienye/dhcp_full_info/mesh/"
    mean_curv_path = "/scratch/hdienye/dhcp_full_info/mean_curv_tex/"
    
    print("=== Mesh and Mean Curvature Correspondence Checker ===\n")
    
    # Check if directories exist
    if not os.path.exists(mesh_path):
        print(f"ERROR: Mesh directory does not exist: {mesh_path}")
        return
    
    if not os.path.exists(mean_curv_path):
        print(f"ERROR: Mean curvature directory does not exist: {mean_curv_path}")
        return
    
    print(f"Mesh directory: {mesh_path}")
    print(f"Mean curvature directory: {mean_curv_path}\n")
    
    # Get all mesh files (smoothed meshes with prefix 'smooth_5_')
    mesh_pattern = os.path.join(mesh_path, "smooth_5_*.surf.gii")
    mesh_files = glob.glob(mesh_pattern)
    
    # Get all mean curvature files (with prefix 'filt_mean_curv_')
    curv_pattern = os.path.join(mean_curv_path, "filt_mean_curv_*.surf.gii")
    curv_files = glob.glob(curv_pattern)
    
    print(f"Found {len(mesh_files)} mesh files")
    print(f"Found {len(curv_files)} mean curvature files\n")
    
    if len(mesh_files) == 0 and len(curv_files) == 0:
        print("WARNING: No files found in either directory. Processing may not have been completed.")
        return
    
    # Extract base filenames
    mesh_bases = set()
    for mesh_file in mesh_files:
        base = extract_base_filename(mesh_file, "smooth_5_")
        mesh_bases.add(base)
    
    curv_bases = set()
    for curv_file in curv_files:
        base = extract_base_filename(curv_file, "filt_mean_curv_")
        curv_bases.add(base)
    
    # Find matches and mismatches
    both_exist = mesh_bases.intersection(curv_bases)
    mesh_only = mesh_bases - curv_bases
    curv_only = curv_bases - mesh_bases
    
    print("=== CORRESPONDENCE ANALYSIS ===")
    print(f"Files with both mesh and curvature: {len(both_exist)}")
    print(f"Files with mesh only: {len(mesh_only)}")
    print(f"Files with curvature only: {len(curv_only)}")
    
    # Calculate statistics
    total_unique = len(mesh_bases.union(curv_bases))
    if total_unique > 0:
        correspondence_rate = (len(both_exist) / total_unique) * 100
        print(f"Correspondence rate: {correspondence_rate:.1f}%\n")
    
    # Report missing correspondences
    if mesh_only:
        print("=== MESHES WITHOUT CORRESPONDING MEAN CURVATURE ===")
        print(f"Found {len(mesh_only)} mesh files without mean curvature:")
        for i, base in enumerate(sorted(mesh_only), 1):
            print(f"{i:3d}. {base}")
        print()
    
    if curv_only:
        print("=== MEAN CURVATURES WITHOUT CORRESPONDING MESH ===")
        print(f"Found {len(curv_only)} mean curvature files without mesh:")
        for i, base in enumerate(sorted(curv_only), 1):
            print(f"{i:3d}. {base}")
        print()
    
    # Create detailed report
    if both_exist or mesh_only or curv_only:
        create_detailed_report(both_exist, mesh_only, curv_only, mesh_path, mean_curv_path)
    
    # Summary
    print("=== SUMMARY ===")
    if len(both_exist) == total_unique and total_unique > 0:
        print("✅ SUCCESS: All files have complete correspondence!")
    elif len(both_exist) > 0:
        print(f"⚠️  PARTIAL: {len(both_exist)}/{total_unique} files have complete correspondence")
        print(f"   Missing: {len(mesh_only)} curvature files, {len(curv_only)} mesh files")
    else:
        print("❌ ERROR: No files have complete correspondence!")
    
    return {
        'both_exist': both_exist,
        'mesh_only': mesh_only,
        'curv_only': curv_only,
        'correspondence_rate': correspondence_rate if total_unique > 0 else 0
    }

def create_detailed_report(both_exist, mesh_only, curv_only, mesh_path, mean_curv_path):
    """
    Create a detailed CSV report of the correspondence analysis.
    """
    report_data = []
    
    # Add complete pairs
    for base in both_exist:
        report_data.append({
            'base_filename': base,
            'has_mesh': True,
            'has_curvature': True,
            'status': 'Complete',
            'mesh_path': os.path.join(mesh_path, f"smooth_5_{base}"),
            'curvature_path': os.path.join(mean_curv_path, f"filt_mean_curv_{base}")
        })
    
    # Add mesh-only files
    for base in mesh_only:
        report_data.append({
            'base_filename': base,
            'has_mesh': True,
            'has_curvature': False,
            'status': 'Missing curvature',
            'mesh_path': os.path.join(mesh_path, f"smooth_5_{base}"),
            'curvature_path': 'MISSING'
        })
    
    # Add curvature-only files
    for base in curv_only:
        report_data.append({
            'base_filename': base,
            'has_mesh': False,
            'has_curvature': True,
            'status': 'Missing mesh',
            'mesh_path': 'MISSING',
            'curvature_path': os.path.join(mean_curv_path, f"filt_mean_curv_{base}")
        })
    
    # Create DataFrame and save report
    if report_data:
        df = pd.DataFrame(report_data)
        df = df.sort_values('base_filename')
        
        report_path = "/scratch/hdienye/dhcp_full_info/mesh_curvature_correspondence_report.csv"
        df.to_csv(report_path, index=False)
        print(f"📄 Detailed report saved to: {report_path}")

def check_file_sizes():
    """
    Additional check to verify file sizes and detect potentially corrupted files.
    """
    print("\n=== FILE SIZE ANALYSIS ===")
    
    mesh_path = "/scratch/hdienye/dhcp_full_info/mesh/"
    mean_curv_path = "/scratch/hdienye/dhcp_full_info/mean_curv_tex/"
    
    # Check mesh file sizes
    mesh_files = glob.glob(os.path.join(mesh_path, "smooth_5_*.surf.gii"))
    curv_files = glob.glob(os.path.join(mean_curv_path, "filt_mean_curv_*.surf.gii.gii"))
    
    def analyze_file_sizes(files, file_type):
        if not files:
            print(f"No {file_type} files found")
            return
        
        sizes = []
        small_files = []
        
        for filepath in files:
            try:
                size = os.path.getsize(filepath)
                sizes.append(size)
                
                # Flag potentially problematic files (less than 1KB)
                if size < 1024:
                    small_files.append((os.path.basename(filepath), size))
            except OSError:
                print(f"Warning: Could not get size for {filepath}")
        
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)
            
            print(f"{file_type} files:")
            print(f"  Count: {len(sizes)}")
            print(f"  Average size: {avg_size/1024:.1f} KB")
            print(f"  Size range: {min_size/1024:.1f} - {max_size/1024:.1f} KB")
            
            if small_files:
                print(f"  ⚠️  {len(small_files)} suspiciously small files:")
                for filename, size in small_files[:5]:  # Show first 5
                    print(f"    {filename}: {size} bytes")
                if len(small_files) > 5:
                    print(f"    ... and {len(small_files) - 5} more")
        print()
    
    analyze_file_sizes(mesh_files, "Mesh")
    analyze_file_sizes(curv_files, "Mean curvature")

def main():
    """
    Main function to run all checks.
    """
    try:
        # Run correspondence check
        results = check_mesh_curvature_correspondence()
        
        # Run file size analysis
        check_file_sizes()
        
        # Provide recommendations
        print("=== RECOMMENDATIONS ===")
        if results and results['correspondence_rate'] < 100:
            print("1. Check the processing logs for errors during mesh smoothing or curvature calculation")
            print("2. Verify that the original surface files exist for missing correspondences")
            print("3. Consider re-running the processing for files with missing correspondences")
            print("4. Check disk space and permissions in output directories")
        else:
            print("✅ All checks passed! Your processing pipeline appears to be working correctly.")
        
    except Exception as e:
        print(f"ERROR: An error occurred during checking: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()