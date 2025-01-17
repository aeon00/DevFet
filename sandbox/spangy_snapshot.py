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

def plot_mesh_with_band_power(vertices, faces, loc_dom_band, band_powers, 
                            camera=None, title=None):
    """
    Plot mesh with colormap based on band powers
    
    :param vertices: numpy array of vertex coordinates
    :param faces: numpy array of face indices
    :param loc_dom_band: numpy array of local dominant band values (0-based indices)
    :param band_powers: dict with band powers
    :param camera: dict with camera position
    :param title: str, plot title
    :return: plotly figure
    """
    # Ensure loc_dom_band values are 1-based to match band numbering
    adjusted_band = loc_dom_band + 1 if np.min(loc_dom_band) == 0 else loc_dom_band
    
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=True,
        hoverinfo='text',
        showscale=True
    )

    # Create colorscale
    colorscale = create_band_power_colormap()
    
    # Create hover text with actual band power values
    hover_text = [f'Band: B{int(b)}\nPower: {band_powers[int(b)]:.2f}' 
                 for b in adjusted_band]
    
    # Update the mesh with color data
    fig_data.update(
        intensity=adjusted_band,
        intensitymode='vertex',
        cmin=1,
        cmax=6,
        colorscale=colorscale,
        colorbar=dict(
            title="Spectral Bands",
            thickness=30,
            len=0.9,
            tickmode='array',
            ticktext=[f'B{i}\n({band_powers[i]:.2f}%)' for i in range(1, 7)],
            tickvals=list(range(1, 7)),
            tickangle=0
        ),
        hovertext=hover_text,
        hoverinfo='text'
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
            camera_medial = dict(
                eye=dict(x=0, y=-2, z=0),    # Camera position from right side
                center=dict(x=0, y=0, z=0),   # Looking at center remains the same
                up=dict(x=-1, y=0, z=0)       # Up vector points in negative x direction
            )

            camera_lateral = dict(
                eye=dict(x=0, y=2, z=0),      # Camera position from left side
                center=dict(x=0, y=0, z=0),    # Looking at center remains the same
                up=dict(x=-1, y=0, z=0)        # Up vector points in negative x direction
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
            fname = f'/home/INT/dienye.h/Python Codes/rough/spangy/snapshots/spectral_bands_dom_lateral{clean_filename}.png'
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
            fname = f'/home/INT/dienye.h/Python Codes/rough/spangy/snapshots/spectral_bands_dom_medial{clean_filename}.png'
            with open(fname, 'wb') as fh:
                fh.write(png_img)





