import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
import os
from slam.differential_geometry import laplacian_mesh_smoothing

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

mesh_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/mesh'
mean_tex_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/mean_curv_tex/'
principal_tex_dir = '/envau/work/meca/users/dienye.h/meso_envau_sync/dhcp_full_info/principal_curv_tex/'

# Ensure output directories exist
ensure_dir_exists(mean_tex_dir)
ensure_dir_exists(principal_tex_dir)

for mesh_file in os.listdir(mesh_dir):
    if not mesh_file.endswith(('.gii', '.mesh', '.ply')):  # Add appropriate mesh extensions
        continue
    
    # Extract filename without extension for texture naming
    filename = os.path.splitext(mesh_file)[0]
    
    # Define the path where the mean curvature texture would be saved
    mean_tex_path = os.path.join(mean_tex_dir, 'filt_mean_curv_{}.gii'.format(filename))
    
    # Check if mean curvature texture already exists
    if os.path.exists(mean_tex_path):
        print(f"Mean curvature texture already exists for {mesh_file}, skipping...")
        continue
    
    print(f"Processing mesh: {mesh_file}")
    
    # Load and process the mesh
    mesh_path = os.path.join(mesh_dir, mesh_file)
    mesh = sio.load_mesh(mesh_path)
    mesh_smooth_5 = laplacian_mesh_smoothing(mesh, nb_iter=5, dt=0.1)
    mesh = mesh_smooth_5
    
    # CURVATURE
    print("compute the mean curvature")
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
        scurv.curvatures_and_derivatives(mesh)
    tex_PrincipalCurvatures = stex.TextureND(PrincipalCurvatures)
    
    # Save principal curvature texture
    principal_tex_path = os.path.join(principal_tex_dir, 'principal_curv_{}.gii'.format(filename))
    sio.write_texture(tex_PrincipalCurvatures, principal_tex_path)
    
    # Compute and save mean curvature texture
    mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
    tex_mean_curv = stex.TextureND(mean_curv)
    tex_mean_curv.z_score_filtering(z_thresh=3)
    
    sio.write_texture(tex_mean_curv, mean_tex_path)
    print(f"Completed processing: {mesh_file}")

print("All meshes processed!")