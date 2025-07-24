#!/usr/bin/env python3
"""
Script to save medial and lateral view snapshots of cortical surface mesh GIFTI files using Nilearn.
This script is for .surf.gii files (geometry) not .func.gii files (functional data).
"""

import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import surface
import numpy as np
import nibabel as nib

def save_surface_mesh_snapshots(surf_gifti_path, output_dir, title_prefix="", 
                                bg_color='white', mesh_color='lightgray'):
    """
    Save medial and lateral view snapshots of a surface mesh GIFTI file.
    
    Parameters:
    -----------
    surf_gifti_path : str
        Path to the surface mesh GIFTI file (.surf.gii)
    output_dir : str
        Directory to save the snapshot images
    title_prefix : str
        Prefix for the saved image filenames
    bg_color : str
        Background color for the plot
    mesh_color : str or None
        Color for the mesh surface
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename for saving
    base_name = Path(surf_gifti_path).stem
    if title_prefix:
        base_name = f"{title_prefix}_{base_name}"
    
    try:
        # Load the surface mesh using nibabel directly
        surf_img = nib.load(surf_gifti_path)
        
        # Extract vertices and faces
        vertices = surf_img.darrays[0].data
        faces = surf_img.darrays[1].data
        
        # Convert faces to integers if they're float (common issue with some GIFTI files)
        if faces.dtype != np.int32 and faces.dtype != np.int64:
            faces = faces.astype(np.int32)
            print(f"Converted face indices from {surf_img.darrays[1].data.dtype} to int32")
        
        print(f"Loaded mesh from {surf_gifti_path}")
        print(f"Vertices shape: {vertices.shape}, dtype: {vertices.dtype}")
        print(f"Faces shape: {faces.shape}, dtype: {faces.dtype}")
        print(f"Face index range: {faces.min()} to {faces.max()}")
        
        # Validate that face indices are within vertex range
        if faces.max() >= vertices.shape[0]:
            print(f"Warning: Face indices exceed vertex count. Max face index: {faces.max()}, Vertices: {vertices.shape[0]}")
        
        # Create the surface mesh tuple that nilearn expects
        surf_mesh = (vertices, faces)
        
        # Determine hemisphere from filename
        if 'left' in surf_gifti_path.lower() or 'lh' in surf_gifti_path.lower():
            hemi = 'left'
        elif 'right' in surf_gifti_path.lower() or 'rh' in surf_gifti_path.lower():
            hemi = 'right'
        else:
            # Try to infer from vertex positions (left hemisphere typically has negative x values)
            if np.mean(vertices[:, 0]) < 0:
                hemi = 'left'
            else:
                hemi = 'right'
            print(f"Inferred hemisphere: {hemi} based on vertex positions")
        
        # Set up plotting parameters for surface mesh
        plot_kwargs = {
            'surf_mesh': surf_mesh,
            'bg_map': None,  # No background statistical map
            'surf_map': None,  # No surface data to map
            'avg_method': 'mean'
        }
        
        # Save lateral view
        fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
        try:
            plotting.plot_surf(
                hemi=hemi,
                view='lateral',
                figure=fig,
                title=f'{base_name} - Lateral View',
                bg_on_data=True,
                darkness=0.7,
                alpha=1.0,
                **plot_kwargs
            )
            
            lateral_path = os.path.join(output_dir, f'{base_name}_lateral.png')
            plt.savefig(lateral_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            print(f"Saved lateral view: {lateral_path}")
        except Exception as e:
            plt.close()
            print(f"Failed to create lateral view: {str(e)}")
        
        # Save medial view
        fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
        try:
            plotting.plot_surf(
                hemi=hemi,
                view='medial', 
                figure=fig,
                title=f'{base_name} - Medial View',
                bg_on_data=True,
                darkness=0.7,
                alpha=1.0,
                **plot_kwargs
            )
            
            medial_path = os.path.join(output_dir, f'{base_name}_medial.png')
            plt.savefig(medial_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            print(f"Saved medial view: {medial_path}")
        except Exception as e:
            plt.close()
            print(f"Failed to create medial view: {str(e)}")
        
        # Optional: Save additional views
        # Dorsal view
        fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
        try:
            plotting.plot_surf(
                hemi=hemi,
                view='dorsal', 
                figure=fig,
                title=f'{base_name} - Dorsal View',
                bg_on_data=True,
                darkness=0.7,
                alpha=1.0,
                **plot_kwargs
            )
            
            dorsal_path = os.path.join(output_dir, f'{base_name}_dorsal.png')
            plt.savefig(dorsal_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            print(f"Saved dorsal view: {dorsal_path}")
        except Exception as e:
            plt.close()
            print(f"Failed to create dorsal view: {str(e)}")
        
        # Ventral view
        fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
        try:
            plotting.plot_surf(
                hemi=hemi,
                view='ventral', 
                figure=fig,
                title=f'{base_name} - Ventral View',
                bg_on_data=True,
                darkness=0.7,
                alpha=1.0,
                **plot_kwargs
            )
            
            ventral_path = os.path.join(output_dir, f'{base_name}_ventral.png')
            plt.savefig(ventral_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
            plt.close()
            print(f"Saved ventral view: {ventral_path}")
        except Exception as e:
            plt.close()
            print(f"Failed to create ventral view: {str(e)}")
        
    except Exception as e:
        print(f"Error processing {surf_gifti_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def process_mesh_directory(input_dir, output_dir, file_pattern="*.surf.gii"):
    """
    Process all surface mesh GIFTI files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing surface mesh GIFTI files
    output_dir : str
        Directory to save snapshots
    file_pattern : str
        Pattern to match surface GIFTI files (default: "*.surf.gii")
    """
    
    # Find all surface GIFTI files
    gifti_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not gifti_files:
        print(f"No surface GIFTI files found in {input_dir} matching pattern {file_pattern}")
        return
    
    print(f"Found {len(gifti_files)} surface GIFTI files to process")
    
    for gifti_file in gifti_files:
        print(f"\nProcessing: {gifti_file}")
        
        save_surface_mesh_snapshots(
            surf_gifti_path=gifti_file, 
            output_dir=output_dir, 
            title_prefix="mesh",
            bg_color='white'
        )

def create_quality_control_grid(input_dir, output_dir, max_subjects=20):
    """
    Create a quality control grid showing multiple subjects in one image.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing surface mesh GIFTI files
    output_dir : str
        Directory to save the QC grid
    max_subjects : int
        Maximum number of subjects to include in the grid
    """
    
    # Find all surface GIFTI files
    gifti_files = glob.glob(os.path.join(input_dir, "*.surf.gii"))
    gifti_files = gifti_files[:max_subjects]  # Limit number of files
    
    if not gifti_files:
        print(f"No surface GIFTI files found in {input_dir}")
        return
    
    # Group by hemisphere and subject
    left_files = [f for f in gifti_files if 'left' in f.lower() or 'lh' in f.lower()]
    right_files = [f for f in gifti_files if 'right' in f.lower() or 'rh' in f.lower()]
    
    # Create grid for left hemisphere
    if left_files:
        n_subjects = len(left_files)
        n_cols = 4  # 4 columns
        n_rows = (n_subjects + n_cols - 1) // n_cols  # Calculate rows needed
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Left Hemisphere - Quality Control', fontsize=16)
        
        for i, gifti_file in enumerate(left_files):
            row = i // n_cols
            col = i % n_cols
            
            try:
                # Load surface
                surf_img = nib.load(gifti_file)
                vertices = surf_img.darrays[0].data
                faces = surf_img.darrays[1].data
                
                # Convert faces to integers if needed
                if faces.dtype != np.int32 and faces.dtype != np.int64:
                    faces = faces.astype(np.int32)
                
                surf_mesh = (vertices, faces)
                
                # Plot on the subplot
                plotting.plot_surf(
                    surf_mesh=surf_mesh,
                    hemi='left',
                    view='lateral',
                    figure=fig,
                    axes=axes[row, col],
                    title=Path(gifti_file).stem.split('_')[1] if '_' in Path(gifti_file).stem else Path(gifti_file).stem,
                    bg_on_data=True,
                    darkness=0.7
                )
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error loading\n{Path(gifti_file).name}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                print(f"Error processing {gifti_file}: {str(e)}")
        
        # Hide unused subplots
        for i in range(len(left_files), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        left_grid_path = os.path.join(output_dir, 'QC_grid_left_hemisphere.png')
        plt.savefig(left_grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved left hemisphere QC grid: {left_grid_path}")

if __name__ == "__main__":
    # Your specific paths based on the error log
    input_directory = "/home/INT/dienye.h/python_files/qc_identified_meshes/dHCP/mesh"
    output_directory = "/home/INT/dienye.h/python_files/qc_identified_meshes/dHCP/snapshots"
    
    # Process all surface mesh files
    process_mesh_directory(input_directory, output_directory, "*.surf.gii")
    
    # Close all matplotlib figures to prevent memory issues
    plt.close('all')
    
    # Optional: Create QC grid
    # create_quality_control_grid(input_directory, output_directory, max_subjects=25)
    
    print("Surface mesh visualization complete!")


# """
# Script to save medial and lateral view snapshots of cortical surface mesh GIFTI files using Nilearn.
# This script is for .surf.gii files (geometry) not .func.gii files (functional data).
# Modified to flip images if they appear upside down.
# """

# import os
# import glob
# from pathlib import Path
# import matplotlib.pyplot as plt
# from nilearn import plotting
# from nilearn import surface
# import numpy as np
# import nibabel as nib
# from PIL import Image

# def save_surface_mesh_snapshots(surf_gifti_path, output_dir, title_prefix="", 
#                                 bg_color='white', mesh_color='lightgray', flip_images=True):
#     """
#     Save medial and lateral view snapshots of a surface mesh GIFTI file.
    
#     Parameters:
#     -----------
#     surf_gifti_path : str
#         Path to the surface mesh GIFTI file (.surf.gii)
#     output_dir : str
#         Directory to save the snapshot images
#     title_prefix : str
#         Prefix for the saved image filenames
#     bg_color : str
#         Background color for the plot
#     mesh_color : str or None
#         Color for the mesh surface
#     flip_images : bool
#         Whether to flip the saved images vertically (for upside-down meshes)
#     """
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get base filename for saving
#     base_name = Path(surf_gifti_path).stem
#     if title_prefix:
#         base_name = f"{title_prefix}_{base_name}"
    
#     try:
#         # Load the surface mesh using nibabel directly
#         surf_img = nib.load(surf_gifti_path)
        
#         # Extract vertices and faces
#         vertices = surf_img.darrays[0].data
#         faces = surf_img.darrays[1].data
        
#         # Convert faces to integers if they're float (common issue with some GIFTI files)
#         if faces.dtype != np.int32 and faces.dtype != np.int64:
#             faces = faces.astype(np.int32)
#             print(f"Converted face indices from {surf_img.darrays[1].data.dtype} to int32")
        
#         print(f"Loaded mesh from {surf_gifti_path}")
#         print(f"Vertices shape: {vertices.shape}, dtype: {vertices.dtype}")
#         print(f"Faces shape: {faces.shape}, dtype: {faces.dtype}")
#         print(f"Face index range: {faces.min()} to {faces.max()}")
        
#         # Validate that face indices are within vertex range
#         if faces.max() >= vertices.shape[0]:
#             print(f"Warning: Face indices exceed vertex count. Max face index: {faces.max()}, Vertices: {vertices.shape[0]}")
        
#         # Create the surface mesh tuple that nilearn expects
#         surf_mesh = (vertices, faces)
        
#         # Determine hemisphere from filename
#         if 'left' in surf_gifti_path.lower() or 'lh' in surf_gifti_path.lower():
#             hemi = 'left'
#         elif 'right' in surf_gifti_path.lower() or 'rh' in surf_gifti_path.lower():
#             hemi = 'right'
#         else:
#             # Try to infer from vertex positions (left hemisphere typically has negative x values)
#             if np.mean(vertices[:, 0]) < 0:
#                 hemi = 'left'
#             else:
#                 hemi = 'right'
#             print(f"Inferred hemisphere: {hemi} based on vertex positions")
        
#         # Set up plotting parameters for surface mesh
#         plot_kwargs = {
#             'surf_mesh': surf_mesh,
#             'bg_map': None,  # No background statistical map
#             'surf_map': None,  # No surface data to map
#             'avg_method': 'mean'
#         }
        
#         # Save lateral view
#         fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
#         try:
#             plotting.plot_surf(
#                 hemi=hemi,
#                 view='lateral',
#                 figure=fig,
#                 title=f'{base_name} - Lateral View',
#                 bg_on_data=True,
#                 darkness=0.7,
#                 alpha=1.0,
#                 **plot_kwargs
#             )
            
#             lateral_path = os.path.join(output_dir, f'{base_name}_lateral.png')
#             plt.savefig(lateral_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
#             plt.close()
            
#             # Flip the image if requested
#             if flip_images:
#                 flip_image_vertically(lateral_path)
                
#             print(f"Saved lateral view: {lateral_path}")
#         except Exception as e:
#             plt.close()
#             print(f"Failed to create lateral view: {str(e)}")
        
#         # Save medial view
#         fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
#         try:
#             plotting.plot_surf(
#                 hemi=hemi,
#                 view='medial', 
#                 figure=fig,
#                 title=f'{base_name} - Medial View',
#                 bg_on_data=True,
#                 darkness=0.7,
#                 alpha=1.0,
#                 **plot_kwargs
#             )
            
#             medial_path = os.path.join(output_dir, f'{base_name}_medial.png')
#             plt.savefig(medial_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
#             plt.close()
            
#             # Flip the image if requested
#             if flip_images:
#                 flip_image_vertically(medial_path)
                
#             print(f"Saved medial view: {medial_path}")
#         except Exception as e:
#             plt.close()
#             print(f"Failed to create medial view: {str(e)}")
        
#         # Optional: Save additional views
#         # Dorsal view
#         fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
#         try:
#             plotting.plot_surf(
#                 hemi=hemi,
#                 view='dorsal', 
#                 figure=fig,
#                 title=f'{base_name} - Dorsal View',
#                 bg_on_data=True,
#                 darkness=0.7,
#                 alpha=1.0,
#                 **plot_kwargs
#             )
            
#             dorsal_path = os.path.join(output_dir, f'{base_name}_dorsal.png')
#             plt.savefig(dorsal_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
#             plt.close()
            
#             # Flip the image if requested
#             if flip_images:
#                 flip_image_vertically(dorsal_path)
                
#             print(f"Saved dorsal view: {dorsal_path}")
#         except Exception as e:
#             plt.close()
#             print(f"Failed to create dorsal view: {str(e)}")
        
#         # Ventral view
#         fig = plt.figure(figsize=(10, 8), facecolor=bg_color)
#         try:
#             plotting.plot_surf(
#                 hemi=hemi,
#                 view='ventral', 
#                 figure=fig,
#                 title=f'{base_name} - Ventral View',
#                 bg_on_data=True,
#                 darkness=0.7,
#                 alpha=1.0,
#                 **plot_kwargs
#             )
            
#             ventral_path = os.path.join(output_dir, f'{base_name}_ventral.png')
#             plt.savefig(ventral_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
#             plt.close()
            
#             # Flip the image if requested
#             if flip_images:
#                 flip_image_vertically(ventral_path)
                
#             print(f"Saved ventral view: {ventral_path}")
#         except Exception as e:
#             plt.close()
#             print(f"Failed to create ventral view: {str(e)}")
        
#     except Exception as e:
#         print(f"Error processing {surf_gifti_path}: {str(e)}")
#         import traceback
#         traceback.print_exc()

# def flip_image_vertically(image_path):
#     """
#     Flip an image vertically but preserve the title text at the top.
    
#     Parameters:
#     -----------
#     image_path : str
#         Path to the image file to flip
#     """
#     try:
#         # Open the image
#         with Image.open(image_path) as img:
#             img_array = np.array(img)
            
#             # Find the title area (usually the top portion with text)
#             # We'll look for the first row that contains non-background content
#             # and assume everything above a certain threshold is title area
            
#             # Convert to grayscale to detect text area
#             if len(img_array.shape) == 3:
#                 gray = np.mean(img_array, axis=2)
#             else:
#                 gray = img_array
            
#             # Find rows that are mostly background (white/light colored)
#             # Assuming background is light colored (high values)
#             row_means = np.mean(gray, axis=1)
            
#             # Find the title boundary - look for the first significant drop in brightness
#             # that indicates we've moved from title area to the 3D rendered area
#             title_height = 0
#             threshold = np.max(row_means) * 0.95  # 95% of max brightness
            
#             for i in range(len(row_means)):
#                 if row_means[i] < threshold:
#                     # Look for a sustained drop (not just a single line)
#                     if i + 10 < len(row_means) and np.mean(row_means[i:i+10]) < threshold:
#                         title_height = i
#                         break
            
#             # If we couldn't detect a clear title area, use a conservative estimate
#             # Typically matplotlib titles take up about 10-15% of the image height
#             if title_height < img_array.shape[0] * 0.05:
#                 title_height = int(img_array.shape[0] * 0.12)
            
#             # Split the image into title and content areas
#             title_area = img_array[:title_height]
#             content_area = img_array[title_height:]
            
#             # Flip only the content area (the 3D mesh visualization)
#             flipped_content = np.flip(content_area, axis=0)
            
#             # Recombine title (unflipped) with flipped content
#             result_array = np.vstack([title_area, flipped_content])
            
#             # Convert back to PIL Image and save
#             result_img = Image.fromarray(result_array.astype(np.uint8))
#             result_img.save(image_path)
            
#         print(f"  -> Flipped image (preserving title): {image_path}")
#     except Exception as e:
#         print(f"  -> Failed to flip image {image_path}: {str(e)}")

# def process_mesh_directory(input_dir, output_dir, file_pattern="*.surf.gii", flip_images=True):
#     """
#     Process all surface mesh GIFTI files in a directory.
    
#     Parameters:
#     -----------
#     input_dir : str
#         Directory containing surface mesh GIFTI files
#     output_dir : str
#         Directory to save snapshots
#     file_pattern : str
#         Pattern to match surface GIFTI files (default: "*.surf.gii")
#     flip_images : bool
#         Whether to flip the saved images vertically
#     """
    
#     # Find all surface GIFTI files
#     gifti_files = glob.glob(os.path.join(input_dir, file_pattern))
    
#     if not gifti_files:
#         print(f"No surface GIFTI files found in {input_dir} matching pattern {file_pattern}")
#         return
    
#     print(f"Found {len(gifti_files)} surface GIFTI files to process")
#     if flip_images:
#         print("Images will be flipped vertically after saving")
    
#     for gifti_file in gifti_files:
#         print(f"\nProcessing: {gifti_file}")
        
#         save_surface_mesh_snapshots(
#             surf_gifti_path=gifti_file, 
#             output_dir=output_dir, 
#             title_prefix="mesh",
#             bg_color='white',
#             flip_images=flip_images
#         )

# def create_quality_control_grid(input_dir, output_dir, max_subjects=20, flip_images=True):
#     """
#     Create a quality control grid showing multiple subjects in one image.
    
#     Parameters:
#     -----------
#     input_dir : str
#         Directory containing surface mesh GIFTI files
#     output_dir : str
#         Directory to save the QC grid
#     max_subjects : int
#         Maximum number of subjects to include in the grid
#     flip_images : bool
#         Whether to flip the final QC grid image
#     """
    
#     # Find all surface GIFTI files
#     gifti_files = glob.glob(os.path.join(input_dir, "*.surf.gii"))
#     gifti_files = gifti_files[:max_subjects]  # Limit number of files
    
#     if not gifti_files:
#         print(f"No surface GIFTI files found in {input_dir}")
#         return
    
#     # Group by hemisphere and subject
#     left_files = [f for f in gifti_files if 'left' in f.lower() or 'lh' in f.lower()]
#     right_files = [f for f in gifti_files if 'right' in f.lower() or 'rh' in f.lower()]
    
#     # Create grid for left hemisphere
#     if left_files:
#         n_subjects = len(left_files)
#         n_cols = 4  # 4 columns
#         n_rows = (n_subjects + n_cols - 1) // n_cols  # Calculate rows needed
        
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
#         if n_rows == 1:
#             axes = axes.reshape(1, -1)
        
#         fig.suptitle('Left Hemisphere - Quality Control', fontsize=16)
        
#         for i, gifti_file in enumerate(left_files):
#             row = i // n_cols
#             col = i % n_cols
            
#             try:
#                 # Load surface
#                 surf_img = nib.load(gifti_file)
#                 vertices = surf_img.darrays[0].data
#                 faces = surf_img.darrays[1].data
                
#                 # Convert faces to integers if needed
#                 if faces.dtype != np.int32 and faces.dtype != np.int64:
#                     faces = faces.astype(np.int32)
                
#                 surf_mesh = (vertices, faces)
                
#                 # Plot on the subplot
#                 plotting.plot_surf(
#                     surf_mesh=surf_mesh,
#                     hemi='left',
#                     view='lateral',
#                     figure=fig,
#                     axes=axes[row, col],
#                     title=Path(gifti_file).stem.split('_')[1] if '_' in Path(gifti_file).stem else Path(gifti_file).stem,
#                     bg_on_data=True,
#                     darkness=0.7
#                 )
                
#             except Exception as e:
#                 axes[row, col].text(0.5, 0.5, f'Error loading\n{Path(gifti_file).name}', 
#                                   ha='center', va='center', transform=axes[row, col].transAxes)
#                 print(f"Error processing {gifti_file}: {str(e)}")
        
#         # Hide unused subplots
#         for i in range(len(left_files), n_rows * n_cols):
#             row = i // n_cols
#             col = i % n_cols
#             axes[row, col].axis('off')
        
#         plt.tight_layout()
#         left_grid_path = os.path.join(output_dir, 'QC_grid_left_hemisphere.png')
#         plt.savefig(left_grid_path, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         # Flip the QC grid image if requested
#         if flip_images:
#             flip_image_vertically(left_grid_path)
            
#         print(f"Saved left hemisphere QC grid: {left_grid_path}")

# if __name__ == "__main__":
#     # Your specific paths based on the error log
#     input_directory = "/home/INT/dienye.h/python_files/qc_identified_meshes/MarsFet/mesh"
#     output_directory = "/home/INT/dienye.h/python_files/qc_identified_meshes/MarsFet/snapshots"
    
#     # Process all surface mesh files with image flipping enabled
#     process_mesh_directory(input_directory, output_directory, "*.surf.gii", flip_images=True)
    
#     # Close all matplotlib figures to prevent memory issues
#     plt.close('all')
    
#     # Optional: Create QC grid (also flipped)
#     # create_quality_control_grid(input_directory, output_directory, max_subjects=25, flip_images=True)
    
#     print("Surface mesh visualization complete!")