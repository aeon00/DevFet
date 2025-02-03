import plotly.graph_objs as go
import numpy as np
import nibabel as nib
import slam.io as sio
import slam.texture as stex
import os
import pandas as pd


import plotly.graph_objs as go
import numpy as np

def read_gii_file(file_path):
    """
    Read scalar data from a GIFTI file
    
    :param file_path: str, path to the GIFTI scalar file
    :return: numpy array of scalar values or None if error
    """
    try:
        gifti_img = nib.load(file_path)
        scalars = gifti_img.darrays[0].data
        return scalars
    except Exception as e:
        print(f"Error loading texture: {e}")
        return None

def create_custom_colormap():
    """
    Create a custom colormap for values from -6 to 6
    Returns both the colormap and the discrete colors for the legend
    """
    # Define colors for negative values (cool colors)
    negative_colors = [
        '#FF0000', # red
        '#00FF00', # green
        '#0000FF', # blue
        '#92c5de',  # very light blue
        '#d1e5f0',  # pale blue
        '#f7f7f7',  # very pale blue for B-1
    ]
    
    # Define colors for positive values (warm colors)
    positive_colors = [
        '#fddbc7',  # pale red
        '#f4a582',  # light red
        '#d6604d',  # medium red
        '#d6604d',  # medium red
        '#b2182b',  # dark red
        '#67001f',  # very dark red
    ]
    
    # Create color mapping for legend
    value_color_map = {}
    for i, color in enumerate(reversed(negative_colors)):
        value_color_map[-(i+1)] = color
    for i, color in enumerate(positive_colors):
        value_color_map[i+1] = color
        
    # Create continuous colorscale
    colorscale = []
    
    # Add negative colors
    for i, color in enumerate(negative_colors):
        pos = i / (len(negative_colors) - 1) * 0.5
        colorscale.append([pos, color])
    
    # Add positive colors
    for i, color in enumerate(positive_colors):
        pos = 0.5 + (i / (len(positive_colors) - 1) * 0.5)
        colorscale.append([pos, color])
    
    return colorscale, value_color_map

def plot_mesh_with_legend(vertices, faces, scalars, view_type='both', selected_bands=None, camera=None, title=None):
    """
    Plot mesh with custom legend instead of colorbar
    
    Parameters:
    - vertices: mesh vertices
    - faces: mesh faces
    - scalars: texture values
    - view_type: 'both', 'positive', or 'negative'
    - camera: camera position dictionary
    - title: plot title
    """
    # Create custom colormap and value-color mapping
    colorscale, value_color_map = create_custom_colormap()
    
    # Create a copy of scalars to modify
    display_scalars = scalars.copy()
    
    # Apply band selection if specified
    if selected_bands is not None:
        # Convert all values not in selected_bands to NaN
        mask = np.zeros_like(display_scalars, dtype=bool)
        for band in selected_bands:
            mask |= (np.round(display_scalars) == band)
        display_scalars[~mask] = np.nan
    
    # Apply view filtering after band selection
    if view_type == 'positive':
        display_scalars[display_scalars < 0] = np.nan
    elif view_type == 'negative':
        display_scalars[display_scalars > 0] = np.nan
    
    # Clip values to our range of interest (-6 to 6)
    display_scalars = np.clip(display_scalars, -6, 6)
    
    # Create the mesh
    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=display_scalars,
        intensitymode='vertex',
        colorscale=colorscale,
        cmin=-6,
        cmax=6,
        showscale=False,  # Hide the colorbar
        hovertemplate='Value: %{intensity:.2f}<extra></extra>'
    )
    
    # Create legend items
    legend_traces = []
    
    # Function to add legend items based on view type
    def add_legend_items(values):
        for val in values:
            color = value_color_map[val]
            legend_traces.append(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'B{val}',
                    showlegend=True
                )
            )
    
    # Add legend items based on selected bands and view type
    if selected_bands is not None:
        # Sort selected bands to maintain legend order
        sorted_bands = sorted(selected_bands)
        add_legend_items([b for b in sorted_bands if 
                         (b < 0 and view_type in ['both', 'negative']) or
                         (b > 0 and view_type in ['both', 'positive'])])
    else:
        # Default behavior if no bands are selected
        if view_type in ['both', 'negative']:
            add_legend_items(range(-6, 0))
        if view_type in ['both', 'positive']:
            add_legend_items(range(1, 7))
    
    # Create the figure
    fig = go.Figure(data=[mesh] + legend_traces)
    
    # Update layout
    layout_dict = dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        height=900,
        width=1000,
        margin=dict(l=10, r=10, b=10, t=50 if title else 10),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    if title:
        layout_dict['title'] = dict(
            text=title,
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        )
    
    fig.update_layout(**layout_dict)
    
    return fig

# Example camera positions (you can modify these as needed)
camera_medial = dict(
    eye=dict(x=2, y=0, z=0),    # Camera position from medial side 
    center=dict(x=0, y=0, z=0),  # Looking at center
    up=dict(x=0, y=0, z=1)       # Up vector points in positive z direction
)

camera_lateral = dict(
    eye=dict(x=-2, y=0, z=0),   # Camera position from lateral side
    center=dict(x=0, y=0, z=0),  # Looking at center
    up=dict(x=0, y=0, z=1)      # Up vector points in positive z direction
)

#Set band values for gyri and sulci

gyri = [4, 5, 6]
sulci = [-6, -5, -4]

directory = "/scratch/gauzias/data/datasets/MarsFet/output/svrtk_BOUNTI/output_BOUNTI_surfaces/haste"  # Add your directory path here
tex_dir = '/scratch/hdienye/spangy/textures'
df = pd.read_csv('/scratch/hdienye/spangy/results/combined_results.csv')

# Collect vertex counts from all meshes
for filename in os.listdir(directory):
    for file in os.listdir(tex_dir):
        # Remove the prefix and keep one .gii
        clean_filename = file.replace("spangy_dom_band_", "").replace(".gii.gii", ".gii")

        if filename == clean_filename:
            participant_session = clean_filename.split('_')[0] + '_' + clean_filename.split('_')[1]
        #Load meshfile
            mesh_file = os.path.join(directory, filename)
            mesh = sio.load_mesh(mesh_file)
            mesh.apply_transform(mesh.principal_inertia_transform)
            vertices = mesh.vertices
            faces = mesh.faces

            # Load generated texture
            tex_file = os.path.join(tex_dir, file)
            loc_dom_band_texture = read_gii_file(tex_file)

            sulci = [-6, -5, -4]
            fig = plot_mesh_with_legend(
                vertices=mesh.vertices,
                faces=mesh.faces,
                scalars=loc_dom_band_texture,
                selected_bands=sulci,  # Will show only sulci bands 4, 5, 6
                camera=camera_lateral,
                title='Negative Bands Visualization'
            )
            fig.write_image(f"/home/INT/dienye.h/Python Codes/rough_hemisphere/spangy_textures/{participant_session}.png")





