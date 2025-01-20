import slam.io as sio
import slam.texture as stex
import slam.curvature as scurv
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.colors as pc
import nibabel as nib
import trimesh
import plotly.io
import slam.spangy as spgy
import time
import plotly
import pandas as pd


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


def create_band_power_colormap():
    """
    Create a discrete colormap for spectral band powers with distinct colors
    """
    colors = [
        '#D3D3D3',  # Grey for B1
        '#D3D3D3',  # Grey for B2
        '#D3D3D3',  # Grey for B3
        '#4B89DC',  # Ocean blue for B4
        '#48CFAD',  # Mint green for B5
        '#EC87C0'   # Rose pink for B6
    ]
    
    colorscale = []
    num_bands = 6
    
    for i in range(num_bands):
        start = i / num_bands
        end = (i + 1) / num_bands
        colorscale.append([start, colors[i]])
        colorscale.append([end, colors[i]])
    
    return colorscale


def calculate_band_percentages(loc_dom_band):
    """
    Calculate the percentage of vertices for each band
    """
    total_vertices = len(loc_dom_band)
    percentages = {}
    
    # Create mapping from actual values to display bands
    value_to_band = {
        -6: 6,
        -5: 5,
        -4: 4,
        -3: 3,
        -2: 2,
        -1: 1
    }
    
    # Initialize all percentages to 0
    for i in range(1, 7):
        percentages[i] = 0.0
    
    # Calculate percentages based on actual values
    unique_vals = np.unique(loc_dom_band)
    for val in unique_vals:
        if 0 > val >= -6:
            count = np.sum(loc_dom_band == val)
            percentage = (count / total_vertices) * 100
            band_num = value_to_band[val]
            percentages[band_num] = percentage
    
    return percentages


def plot_mesh_with_band_power(vertices, faces, loc_dom_band, vertex_percentages, 
                            camera=None, title=None):
    """
    Plot mesh with colormap based on band powers
    """
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=True,
        showscale=True
    )

    colorscale = create_band_power_colormap()
    
    fig_data.update(
        intensity=loc_dom_band,
        intensitymode='vertex',
        cmin=-6,
        cmax=-1,
        colorscale=colorscale,
        colorbar=dict(
            title="Spectral Bands",
            thickness=30,
            len=0.9,
            tickmode='array',
            ticktext=[f'B{i}\n({vertex_percentages[i]:.1f}%)' for i in range(1, 7)],
            tickvals=list(range(-6, 0)),
            tickangle=0,
            tickfont=dict(size=12)
        )
    )

    fig = go.Figure(data=[go.Mesh3d(**fig_data)])

    layout_dict = dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        height=900,
        width=1000,
        margin=dict(l=10, r=10, b=10, t=50 if title else 10)
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


def visualize_brain_bands(vertices, faces, loc_dom_band, grouped_spectrum, camera_position):
    """
    Visualize brain bands
    """
    vertex_percentages = calculate_band_percentages(loc_dom_band)
    
    fig = plot_mesh_with_band_power(
        vertices=vertices,
        faces=faces,
        loc_dom_band=loc_dom_band,
        vertex_percentages=vertex_percentages,
        camera=camera_position,
        title="Spectral Band Distribution"
    )
    
    return fig



directory = "/home/INT/dienye.h/Python Codes/rough_surf/"  # Add your directory path here
tex_dir = '/home/INT/dienye.h/Python Codes/rough/spangy/textures/'
df = pd.read_csv('/home/INT/dienye.h/Python Codes/rough/results/surface_analysis_results.csv')

# Collect vertex counts from all meshes
for filename in os.listdir(directory):
    for file in os.listdir(tex_dir):
        # Remove the prefix and keep one .gii
        clean_filename = file.replace("spangy_dom_band_", "").replace(".gii.gii", ".gii")

        if filename == clean_filename:
            participant_session = clean_filename.split('_')[0] + '_' + clean_filename.split('_')[1]
            B1 = df[df['participant_session'] == participant_session]['band_power_B1'].values[0]
            B2 = df[df['participant_session'] == participant_session]['band_power_B2'].values[0]
            B3 = df[df['participant_session'] == participant_session]['band_power_B3'].values[0]
            B4 = df[df['participant_session'] == participant_session]['band_power_B4'].values[0]
            B5 = df[df['participant_session'] == participant_session]['band_power_B5'].values[0]
            B6 = df[df['participant_session'] == participant_session]['band_power_B6'].values[0]
        #Load meshfile
            mesh_file = os.path.join(directory, filename)
            mesh = sio.load_mesh(mesh_file)
            mesh.apply_transform(mesh.principal_inertia_transform)
            vertices = mesh.vertices
            faces = mesh.faces

            # WHOLE BRAIN MEAN-CURVATURE SPECTRUM
            grouped_spectrum = [B1, B2, B3, B4, B5, B6]
            # Load generated texture
            tex_file = os.path.join(tex_dir, file)
            loc_dom_band_texture = read_gii_file(tex_file)

            # Set cameras for snapshots
            camera_superior_medial = dict(
                eye=dict(x=0, y=-1.4, z=1.4),  # Camera position up and right
                center=dict(x=0, y=0, z=0),     # Looking at center
                up=dict(x=-1, y=0, z=0)         # Up vector points in negative x direction
            )

            camera_medial = dict(
                eye=dict(x=0, y=-2, z=0),       # Camera position straight from right
                center=dict(x=0, y=0, z=0),     # Looking at center
                up=dict(x=-1, y=0, z=0)         # Up vector points in negative x direction
            )

            camera_superior_lateral = dict(
                eye=dict(x=0, y=1.4, z=1.4),    # Camera position up and left
                center=dict(x=0, y=0, z=0),     # Looking at center
                up=dict(x=-1, y=0, z=0)         # Up vector points in negative x direction
            )

            camera_lateral = dict(
                eye=dict(x=0, y=2, z=0),        # Camera position straight from left
                center=dict(x=0, y=0, z=0),     # Looking at center
                up=dict(x=-1, y=0, z=0)         # Up vector points in negative x direction
            )
            camera_superior = dict(
                eye=dict(x=0, y=0, z=2),    # Camera position from above
                center=dict(x=0, y=0, z=0),  # Looking at center
                up=dict(x=0, y=1, z=0)      # Up vector points in positive y direction
            )

            camera_inferior = dict(
                eye=dict(x=0, y=0, z=-2),   # Camera position from below
                center=dict(x=0, y=0, z=0),  # Looking at center
                up=dict(x=0, y=1, z=0)      # Up vector points in positive y direction
            )

    # Lateral view snapshots
            fig = visualize_brain_bands(
                vertices=vertices,
                faces=faces,
                loc_dom_band=loc_dom_band_texture,
                grouped_spectrum= grouped_spectrum,
                camera_position=camera_superior_medial  # or camera_medial
            )

            # Save the figure
            png_img = plotly.io.to_image(fig, 'png')
            fname = f'/home/INT/dienye.h/Python Codes/rough/spangy/snapshots/spectral_bands_dom_lateral{clean_filename}.png'
            with open(fname, 'wb') as fh:
                fh.write(png_img)

    # Medial view snapshots
            fig = visualize_brain_bands(
                vertices=vertices,
                faces=faces,
                loc_dom_band=loc_dom_band_texture,
                grouped_spectrum= grouped_spectrum,
                camera_position=camera_superior_lateral  # or camera_medial
            )

            # Save the figure
            png_img = plotly.io.to_image(fig, 'png')
            fname = f'/home/INT/dienye.h/Python Codes/rough/spangy/snapshots/spectral_bands_dom_medial{clean_filename}.png'
            with open(fname, 'wb') as fh:
                fh.write(png_img)





