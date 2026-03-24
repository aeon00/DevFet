import numpy as np
import nibabel as nib
import trimesh
from scipy.sparse.linalg import eigsh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import slam.differential_geometry as sdg

# ==========================================
# 1. SPANGY FUNCTIONS (From your snippet)
# ==========================================

def eigenpairs(mesh, nb_eig):
    """
    Compute nb_eig eigen pairs (eigen value and associated eigenvector) of
    the Laplace-Beltrami operator defined on the input mesh.
    """
    print(f"   Computing {nb_eig} eigenfunctions... (this may take a moment)")
    # 'fem' = Finite Element Method, standard for SPANGY
    lap, lap_b = sdg.compute_mesh_laplacian(mesh, lap_type='fem')
    
    # Calculate eigenvalues and eigenvectors
    # sigma=1e-6 helps find the smallest eigenvalues (close to 0) efficiently
    eig_val, eig_vec = eigsh(lap.tocsr(), nb_eig, M=lap_b.tocsr(),
                             sigma=1e-6, which='LM')
    return eig_val, eig_vec, lap_b.tocsr()

# ==========================================
# 2. HELPER FUNCTIONS (IO & VIZ)
# ==========================================

def load_gifti_mesh(mesh_path):
    """Loads a GIFTI mesh file and converts it to a trimesh object."""
    print(f"Loading mesh: {mesh_path}")
    gii = nib.load(mesh_path)
    # Extract vertices (point coordinates) and faces (triangles)
    vertices = gii.agg_data('pointset')
    faces = gii.agg_data('triangle')
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

def load_gifti_texture(texture_path):
    """Loads a GIFTI texture file (scalar map like curvature)."""
    print(f"Loading texture: {texture_path}")
    gii = nib.load(texture_path)
    # Assumes the data is in the first data array
    return gii.darrays[0].data

def create_mesh_trace(mesh, intensity, title, visible=True, colorscale='Jet', cmin=None, cmax=None):
    """Creates a Plotly Mesh3d trace with custom range."""
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        intensity=intensity,
        colorscale=colorscale,
        cmin=cmin,  # <--- Set lower bound
        cmax=cmax,  # <--- Set upper bound
        showscale=False,
        name=title,
        visible=visible,
        hoverinfo='name'
    )

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # --- CONFIGURATION: UPDATE THESE PATHS ---
    MESH_FILE = '/home/INT/dienye.h/python_files/rough/smooth_5_sub-0858_ses-0995_reo-SVR-output-brain-mask-brain_bounti-white.right.surf.gii'       # e.g., 'lh.pial.gii' or 'lh.white.gii'
    TEXTURE_FILE = '/home/INT/dienye.h/python_files/rough/smooth_5_filt_mean_curv_sub-0858_ses-0995_reo-SVR-output-brain-mask-brain_bounti-white.right.surf.gii.gii' # e.g., 'lh.curv.gii' or 'lh.sulc.gii'
    NUM_EIGENMODES = 200              # Total modes to compute
    
    # Indices to visualize (0 is usually constant/DC, so we start from 1)
    # We pick a range from Low Frequency (Global Shape) to High Frequency (Fine Folds)
    # Note: Python uses 0-based indexing, so index 1 is the 2nd mode.
    VIZ_INDICES = [1, 25, 50, 100, 199] 

    try:
        # 1. Load Data
        mesh = load_gifti_mesh(MESH_FILE)
        mesh.vertices[:, [1, 2]] *= -1
        
        try:
            texture_data = load_gifti_texture(TEXTURE_FILE)
            has_texture = True
        except Exception as e:
            print(f"Warning: Could not load texture ({e}). Proceeding with mesh only.")
            has_texture = False

        # 2. Compute Eigenfunctions
        eig_vals, eig_vecs, mass_mat = eigenpairs(mesh, NUM_EIGENMODES)
        print("   Computation complete.")

        # 3. Visualization with Plotly
        print("Generating Plotly visualization...")
        
        # Define subplot titles
        titles = ["Mean Curvature"] if has_texture else ["Mesh Only"]
        titles += [f"Eigenfunction {idx} (λ={eig_vals[idx]:.4f})" for idx in VIZ_INDICES]
        
        # Create figure with 2 rows and 3 columns (total 6 plots)
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'mesh3d'}]*3]*2,
            subplot_titles=titles,
            vertical_spacing=0.05,
            horizontal_spacing=0.02
        )

# Plot 1: The Input Texture
        if has_texture:
            # Calculate robust limits (ignoring the top/bottom 2% of outliers)
            # This fixes the "all white" issue by shrinking the color range
            v_min, v_max = np.percentile(texture_data, [2, 98])
            
            # If the data is centered at 0 (Z-scored), make the limits symmetric
            # so 0 is perfectly white
            limit = max(abs(v_min), abs(v_max))
            v_min = -limit
            v_max = limit

            print(f"   Texture range adjusted to: {v_min:.3f} to {v_max:.3f}")

            fig.add_trace(
                create_mesh_trace(
                    mesh, 
                    texture_data, 
                    "Mean Curvature", 
                    colorscale='RdBu', 
                    cmin=v_min, 
                    cmax=v_max
                ), 
                row=1, col=1
            )
        else:
            # If no texture, just show geometry in grey
            fig.add_trace(go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                                    i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                                    color='lightgrey', name="Geometry"), row=1, col=1)

        # Plot 2-6: The Chosen Eigenfunctions
        # Locations in the 2x3 grid (flattened logic for remaining 5 spots)
        grid_locs = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        
        for i, idx in enumerate(VIZ_INDICES):
            row, col = grid_locs[i]
            eigenmode_data = eig_vecs[:, idx]
            
            trace = create_mesh_trace(
                mesh, 
                eigenmode_data, 
                f"Eigenfunction {idx}"
            )
            fig.add_trace(trace, row=row, col=col)

        # Layout adjustments
        fig.update_layout(
            title_text=f"SPANGY Spectral Decomposition: {MESH_FILE}",
            height=900,
            width=1400,
            scene_camera=dict(up=dict(x=0, y=0, z=-1), eye=dict(x=1.5, y=0, z=0)), # Default view angle
        )
        
        # Remove axes for cleaner "brain" look
        no_axis = dict(showbackground=False, showgrid=False, showline=False, 
                       showticklabels=False, zeroline=False, title='')
        
        for i in range(1, 7):
            # Plotly names scenes as scene, scene2, scene3...
            scene_name = 'scene' if i == 1 else f'scene{i}'
            fig.layout[scene_name].update(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis)

        fig.show()
        print("Visualization launched in your browser.")

    except FileNotFoundError:
        print("Error: Please check your filenames (MESH_FILE / TEXTURE_FILE) at the top of the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")