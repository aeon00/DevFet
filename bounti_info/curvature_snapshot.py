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

def plot_mesh_with_colorbar(vertices, faces, scalars=None, camera=None, 
                          show_contours=False, colormap='jet', 
                          use_black_intervals=False, title=None):
    """
    Modified to use fixed color range from -5 to 5
    """
    fig_data = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        flatshading=False, hoverinfo='text', showscale=False
    )

    if scalars is not None:
        # Set fixed color range from -5 to 5
        color_min, color_max = -5, 5
        
        # Clip the scalar values to the fixed range
        clipped_scalars = np.clip(scalars, color_min, color_max)

        if use_black_intervals:
            colorscale = create_colormap_with_black_stripes(colormap)
        else:
            colorscale = colormap

        fig_data.update(
            intensity=clipped_scalars,
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
            hovertext=[f'Scalar value: {s:.2f} (Original: {orig:.2f})' 
                      for s, orig in zip(clipped_scalars, scalars)]
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

directory = '/home/INT/dienye.h/Python Codes/Surfaces/'  # Add your directory path here
cnt = 0

# Main processing loop
for filename in os.listdir(directory)[:10]:
    if filename.endswith('surf.gii'): 
        mesh_file = os.path.join(directory, filename)
        mesh = sio.load_mesh(mesh_file)
        mesh.apply_transform(mesh.principal_inertia_transform)
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Compute curvatures
        PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
            scurv.curvatures_and_derivatives(mesh)
        
        gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
        mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
        shapeIndex, curvedness = scurv.decompose_curvature(PrincipalCurvatures)

        cnt += 1

        # Save computed maps
        mean_curv_path = f"/home/INT/dienye.h/Python Codes/Curvature/mean curvature/mean_curv_{cnt}.gii"
        gaussian_curv_path = f"/home/INT/dienye.h/Python Codes/Curvature/gaussian curvature/gaussian_curv_{cnt}.gii"
        shape_index_curv_path = f"/home/INT/dienye.h/Python Codes/Curvature/shape index curvature/shape_index_{cnt}.gii"
        
        # Apply z-score filtering and save textures
        for data, path in [(mean_curv, mean_curv_path), 
                          (gaussian_curv, gaussian_curv_path),
                          (shapeIndex, shape_index_curv_path)]:
            tmp_tex = stex.TextureND(data)
            tmp_tex.z_score_filtering(z_thresh=3)
            sio.write_texture(tmp_tex, path)

        # Load generated textures
        mean_curv_texture = read_gii_file(mean_curv_path)
        gaussian_curv_texture = read_gii_file(gaussian_curv_path)
        shape_index_curv_texture = read_gii_file(shape_index_curv_path)

        # Camera positions
        camera_lateral = dict(
            eye=dict(x=-2, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=-2, y=0, z=0)
        )

        camera_medial = dict(
            eye=dict(x=2, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=2, y=0, z=0)
        )

        # Generate plots for both views
        for camera, view in [(camera_lateral, 'lateral'), (camera_medial, 'medial')]:
            for data, name, title in [
                (mean_curv_texture, 'mean_curv', 'Mean Curvature'),
                (gaussian_curv_texture, 'gaussian_curv', 'Gaussian Curvature'),
                (shape_index_curv_texture, 'shape_index_curv', 'Shape Index Curvature')
            ]:
                fig = plot_mesh_with_colorbar(
                    vertices, 
                    faces, 
                    scalars=data,
                    camera=camera,
                    show_contours=False,
                    colormap='jet',
                    use_black_intervals=False,
                    title=title
                )

                png_img = plotly.io.to_image(fig, 'png')
                fname = f'/home/INT/dienye.h/Python Codes/Curvature/{name.replace("_", " ")}/Plots/{name}_{view}_{cnt}.png'
                with open(fname, 'wb') as fh:
                    fh.write(png_img)