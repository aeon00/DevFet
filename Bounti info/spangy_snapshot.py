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

def convert_rgb_to_hex_if_needed(colormap):
    """
    Convert RGB colors to hexadecimal if needed
    
    :param colormap: list of color strings
    :return: list of hexadecimal color strings
    """
    hex_colormap = []
    for color in colormap:
        if color.startswith('rgb'):
            rgb_values = [int(c) for c in color[4:-1].split(',')]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
            hex_colormap.append(hex_color)
        else:
            hex_colormap.append(color)
    return hex_colormap

def create_colormap_with_black_stripes(base_colormap, num_intervals=10, black_line_width=0.01):
    temp_c = pc.get_colorscale(base_colormap)
    temp_c_2 = [ii[1] for ii in temp_c]
    old_colormap = convert_rgb_to_hex_if_needed(temp_c_2)
    custom_colormap = []
    base_intervals = np.linspace(0, 1, len(old_colormap))

    for i in range(len(old_colormap) - 1):
        custom_colormap.append([base_intervals[i], old_colormap[i]])
        if i % (len(old_colormap) // num_intervals) == 0:
            black_start = base_intervals[i]
            black_end = min(black_start + black_line_width, 1)
            custom_colormap.append([black_start, 'rgb(0, 0, 0)'])
            custom_colormap.append([black_end, old_colormap[i]])
    custom_colormap.append([1, old_colormap[-1]])
    return custom_colormap

def plot_mesh_with_colorbar(vertices, faces, scalars=None, color_min=None, color_max=None, 
                          camera=None, show_contours=False, colormap='jet', 
                          use_black_intervals=False, center_colormap_on_zero=False, 
                          title=None):  # Added title parameter
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=False, hoverinfo='text', showscale=False
    )

    if scalars is not None:
        color_min = color_min if color_min is not None else np.min(scalars)
        color_max = color_max if color_max is not None else np.max(scalars)

        if center_colormap_on_zero:
            max_abs_value = max(abs(color_min), abs(color_max))
            color_min, color_max = -max_abs_value, max_abs_value

        if use_black_intervals:
            colorscale = create_colormap_with_black_stripes(colormap)
        else:
            colorscale = colormap

        fig_data.update(
            intensity=scalars,
            intensitymode='vertex',
            cmin=color_min,
            cmax=color_max,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Scalars",
                tickformat=".2f",
                thickness=30,
                len=0.9
            ),
            hovertext=[f'Scalar value: {s:.2f}' for s in scalars]
        )

    fig = go.Figure(data=[go.Mesh3d(**fig_data)])
    if show_contours:
        fig.data[0].update(contour=dict(show=True, color='black', width=2))

    # Update layout to include title if provided
    layout_dict = dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        height=900,
        width=1000,
        margin=dict(l=10, r=10, b=10, t=50 if title else 10)  # Increased top margin if title present
    )
    
    if title:
        layout_dict['title'] = dict(
            text=title,
            x=0.5,  # Center the title
            y=0.95,  # Position from top
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        )
    
    fig.update_layout(**layout_dict)

    return fig


def create_band_based_colormap(band_labels=['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6'], colormap='jet'):
    """
    Create a discrete colormap based on spectral bands
    
    :param band_labels: list of band names
    :param colormap: str, name of the base colormap
    :return: list of [position, color] pairs
    """
    num_bands = len(band_labels)
    
    # Get base colormap
    temp_c = pc.get_colorscale(colormap)
    temp_c_2 = [ii[1] for ii in temp_c]
    base_colors = convert_rgb_to_hex_if_needed(temp_c_2)
    
    # Create discrete intervals for each band
    discrete_colormap = []
    for i in range(num_bands):
        # Calculate position for current band
        pos_start = i / num_bands
        pos_end = (i + 1) / num_bands
        
        # Select color from base colormap
        color_index = int((i / num_bands) * (len(base_colors) - 1))
        color = base_colors[color_index]
        
        # Add color stops to create discrete effect
        discrete_colormap.append([pos_start, color])
        discrete_colormap.append([pos_end, color])
        
        # Add tiny gap between colors (optional)
        if i < num_bands - 1:
            discrete_colormap.append([pos_end, color])
            
    return discrete_colormap

def plot_mesh_with_band_colorbar(vertices, faces, scalars=None, grouped_spectrum=None,
                               camera=None, show_contours=False, colormap='jet', 
                               title=None):
    """
    Plot mesh with colorbar based on spectral bands
    
    :param vertices: numpy array of vertex coordinates
    :param faces: numpy array of face indices
    :param scalars: numpy array of scalar values
    :param grouped_spectrum: numpy array of spectral band values
    :param camera: dict with camera position
    :param show_contours: bool, whether to show contours
    :param colormap: str, colormap name
    :param title: str, plot title
    :return: plotly figure
    """
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=True,
        hoverinfo='text',
        showscale=False
    )

    if scalars is not None:
        # Create band intervals based on grouped_spectrum
        band_values = np.arange(len(grouped_spectrum))
        
        # Create discrete colorscale
        colorscale = create_band_based_colormap(
            band_labels=[f'B{i}' for i in range(len(grouped_spectrum))],
            colormap=colormap
        )

        # Discretize the scalar values into bands
        discretized_scalars = np.zeros_like(scalars)
        for i in range(len(grouped_spectrum)):
            if i == 0:
                mask = (scalars <= grouped_spectrum[0])
            elif i == len(grouped_spectrum) - 1:
                mask = (scalars > grouped_spectrum[-2])
            else:
                mask = (scalars > grouped_spectrum[i-1]) & (scalars <= grouped_spectrum[i])
            discretized_scalars[mask] = i

        fig_data.update(
            intensity=discretized_scalars,
            intensitymode='vertex',
            cmin=0,
            cmax=len(grouped_spectrum) - 1,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title="Spectral Bands",
                thickness=30,
                len=0.9,
                tickmode='array',
                ticktext=[f'B{i}' for i in range(len(grouped_spectrum))],
                tickvals=np.arange(len(grouped_spectrum)),
                tickangle=0
            ),
            hovertext=[f'Band: B{int(s)}, Value: {grouped_spectrum[int(s)]:.2f}' 
                      for s in discretized_scalars]
        )

    fig = go.Figure(data=[go.Mesh3d(**fig_data)])
    if show_contours:
        fig.data[0].update(contour=dict(show=True, color='black', width=2))

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


def create_band_power_colormap():
    """
    Create a discrete colormap based on spectral band powers
    """
    # Define colors for each band - using a more distinctive color scheme
    band_colors = {
        'B4': '#0000FF',  # Blue
        'B5': '#FFFF00',  # Yellow
        'B6': '#FF0000'   # Red
    }
    
    colorscale = []
    for i, (band, color) in enumerate(band_colors.items()):
        pos = i / 2  # Divide by (number of bands - 1)
        colorscale.append([pos, color])
    
    return colorscale

def plot_mesh_with_band_power(vertices, faces, loc_dom_band, band_powers, 
                            camera=None, title=None):
    """
    Plot mesh with colormap based on band powers
    
    :param vertices: numpy array of vertex coordinates
    :param faces: numpy array of face indices
    :param loc_dom_band: numpy array of local dominant band values
    :param band_powers: dict with band powers
    :param camera: dict with camera position
    :param title: str, plot title
    :return: plotly figure
    """
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=True,
        hoverinfo='text',
        showscale=True
    )

    # Create colorscale
    colorscale = create_band_power_colormap()
    
    # Update the mesh with color data
    fig_data.update(
        intensity=loc_dom_band,
        intensitymode='vertex',
        cmin=0,
        cmax=6,  # For bands B0-B6
        colorscale=colorscale,
        colorbar=dict(
            title="Spectral Bands",
            thickness=30,
            len=0.9,
            tickmode='array',
            ticktext=[f'B{i}\n({band_powers[i]:.2f})' for i in range(7)],
            tickvals=list(range(7)),
            tickangle=0
        ),
        # hovertext=[f'Band: B{int(v)}\nPower: {band_powers[int(v)]:.2f}' 
        #           for v in loc_dom_band]
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
    # Define band powers based on your data
    band_powers = {
        1: grouped_spectrum[0],   # B4
        2: grouped_spectrum[1],   # B5
        3: grouped_spectrum[2]    # B6
    }
    
    fig = plot_mesh_with_band_power(
        vertices=vertices,
        faces=faces,
        loc_dom_band=loc_dom_band,
        band_powers=band_powers,
        camera=camera_position,
        title="Spectral Band Distribution"
    )
    
    return fig



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
            grouped_spectrum = [B4, B5, B6]
            # Load generated texture
            tex_file = os.path.join(tex_dir, file)
            loc_dom_band_texture = read_gii_file(tex_file)

            # Set cameras for snapshots
            camera_lateral = dict(
                eye=dict(x=-2, y=0, z=0),  # Position of the camera
                center=dict(x=0, y=0, z=0),     # Point the camera is looking at
                up=dict(x=-2, y=0, z=0)          # Up direction
            )

            camera_medial = dict(
                eye=dict(x=2, y=0, z=0),  # Position of the camera
                center=dict(x=0, y=0, z=0),     # Point the camera is looking at
                up=dict(x=2, y=0, z=0)          # Up direction
            )


    # Lateral view snapshots
            fig = visualize_brain_bands(
                vertices=vertices,
                faces=faces,
                loc_dom_band=loc_dom_band_texture,
                grouped_spectrum= grouped_spectrum,
                camera_position=camera_lateral  # or camera_medial
            )

            # Save the figure
            png_img = plotly.io.to_image(fig, 'png')
            fname = f'/scratch/hdienye/spangy/spangy_snapshots/spectral_bands_dom_lateral{clean_filename}.png'
            with open(fname, 'wb') as fh:
                fh.write(png_img)

    # Medial view snapshots
            fig = visualize_brain_bands(
                vertices=vertices,
                faces=faces,
                loc_dom_band=loc_dom_band_texture,
                grouped_spectrum= grouped_spectrum,
                camera_position=camera_medial  # or camera_medial
            )

            # Save the figure
            png_img = plotly.io.to_image(fig, 'png')
            fname = f'/scratch/hdienye/spangy/spangy_snapshots/spectral_bands_dom_medial{clean_filename}.png'
            with open(fname, 'wb') as fh:
                fh.write(png_img)





