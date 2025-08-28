import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import nibabel as nib
import slam.io as sio
import slam.texture as stex
import os
import pandas as pd
from pathlib import Path
import json

class InteractiveMeshViewer:
    def __init__(self):
        self.current_mesh = None
        self.current_texture = None
        self.current_vertices = None
        self.current_faces = None
        self.mesh_info = {}
        
        # Initialize Dash app with built-in styling
        self.app = dash.Dash(__name__)
        self.app.title = "Interactive 3D Mesh & Texture Viewer"
        
        # Set up the layout
        self.setup_layout()
        self.setup_callbacks()
    
    def read_gii_file(self, file_path):
        """Read scalar data from a GIFTI file"""
        try:
            gifti_img = nib.load(file_path)
            if hasattr(gifti_img, 'darrays') and len(gifti_img.darrays) > 0:
                scalars = gifti_img.darrays[0].data
                return scalars
            else:
                print(f"No data arrays found in {file_path}")
                return None
        except Exception as e:
            print(f"Error loading texture: {e}")
            return None
    
    def create_custom_colormap(self):
        """Create custom colormap for values from -6 to 6"""
        negative_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#92c5de', '#d1e5f0', '#f7f7f7'
        ]
        positive_colors = [
            '#fddbc7', '#f4a582', '#d6604d', '#d6604d', '#b2182b', '#67001f'
        ]
        
        # Create value-color mapping
        value_color_map = {}
        for i, color in enumerate(reversed(negative_colors)):
            value_color_map[-(i+1)] = color
        for i, color in enumerate(positive_colors):
            value_color_map[i+1] = color
            
        # Create continuous colorscale
        colorscale = []
        for i, color in enumerate(negative_colors):
            pos = i / (len(negative_colors) - 1) * 0.5
            colorscale.append([pos, color])
        for i, color in enumerate(positive_colors):
            pos = 0.5 + (i / (len(positive_colors) - 1) * 0.5)
            colorscale.append([pos, color])
        
        return colorscale, value_color_map
    
    def mesh_orientation(self, mesh, hemisphere):
        """Orient mesh for proper visualization"""
        hemisphere = str(hemisphere).lower()
        
        if hemisphere == 'right':
            transfo_180 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            mesh.apply_transform(transfo_180)
            
            camera_lateral = dict(
                eye=dict(x=2, y=0, z=0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=-1)
            )
            camera_medial = dict(
                eye=dict(x=-2, y=0, z=0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=-1)
            )
            
        elif hemisphere == 'left':
            transfo_180 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            mesh.apply_transform(transfo_180)
            
            camera_medial = dict(
                eye=dict(x=2, y=0, z=0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=-1)
            )
            camera_lateral = dict(
                eye=dict(x=-2, y=0, z=0),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=-1)
            )
        
        return mesh, camera_medial, camera_lateral
    
    def setup_layout(self):
        """Set up the Dash app layout with custom CSS styling"""
        
        # Custom CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
                        color: #ffffff;
                        margin: 0;
                        padding: 20px;
                    }
                    
                    .main-container {
                        display: flex;
                        gap: 20px;
                        height: 95vh;
                    }
                    
                    .control-panel {
                        width: 350px;
                        background: rgba(30, 30, 46, 0.95);
                        border-radius: 12px;
                        padding: 20px;
                        overflow-y: auto;
                        border: 2px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
                    }
                    
                    .visualization-panel {
                        flex: 1;
                        background: rgba(30, 30, 46, 0.95);
                        border-radius: 12px;
                        padding: 20px;
                        border: 2px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
                    }
                    
                    .section {
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 20px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }
                    
                    .section h3 {
                        color: #64b5f6;
                        margin-top: 0;
                        margin-bottom: 15px;
                        font-size: 16px;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }
                    
                    .btn {
                        background: linear-gradient(135deg, #4f46e5, #7c3aed);
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 8px 16px;
                        margin: 4px;
                        cursor: pointer;
                        font-size: 12px;
                        font-weight: 600;
                        transition: all 0.3s ease;
                    }
                    
                    .btn:hover {
                        background: linear-gradient(135deg, #5b52e8, #8b47ed);
                        transform: translateY(-2px);
                        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.3);
                    }
                    
                    .btn-success {
                        background: linear-gradient(135deg, #10b981, #059669);
                    }
                    
                    .btn-success:hover {
                        background: linear-gradient(135deg, #34d399, #10b981);
                    }
                    
                    .btn-info {
                        background: linear-gradient(135deg, #06b6d4, #0891b2);
                    }
                    
                    .btn-warning {
                        background: linear-gradient(135deg, #f59e0b, #d97706);
                    }
                    
                    .btn-danger {
                        background: linear-gradient(135deg, #ef4444, #dc2626);
                    }
                    
                    .btn-secondary {
                        background: linear-gradient(135deg, #6b7280, #4b5563);
                    }
                    
                    input, select {
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 6px;
                        color: white;
                        padding: 8px;
                        width: 100%;
                        box-sizing: border-box;
                        margin: 5px 0;
                    }
                    
                    input::placeholder {
                        color: rgba(255, 255, 255, 0.5);
                    }
                    
                    .info-panel {
                        background: rgba(59, 130, 246, 0.1);
                        border: 1px solid rgba(59, 130, 246, 0.3);
                        border-radius: 8px;
                        padding: 12px;
                        margin: 10px 0;
                        color: #d1d5db;
                        font-size: 14px;
                    }
                    
                    .error-panel {
                        background: rgba(239, 68, 68, 0.1);
                        border: 1px solid rgba(239, 68, 68, 0.3);
                        border-radius: 8px;
                        padding: 12px;
                        margin: 10px 0;
                        color: #f87171;
                    }
                    
                    .success-panel {
                        background: rgba(16, 185, 129, 0.1);
                        border: 1px solid rgba(16, 185, 129, 0.3);
                        border-radius: 8px;
                        padding: 12px;
                        margin: 10px 0;
                        color: #34d399;
                    }
                    
                    .checklist-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 8px;
                        margin: 10px 0;
                    }
                    
                    .checklist-item {
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 6px;
                        padding: 4px 8px;
                        font-size: 12px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }
                    
                    h1 {
                        text-align: center;
                        color: #00d4aa;
                        margin-bottom: 30px;
                        font-size: 28px;
                    }
                    
                    label {
                        color: #d1d5db;
                        font-weight: 500;
                        margin-bottom: 5px;
                        display: block;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
        
        self.app.layout = html.Div([
            html.H1("🧠 Interactive 3D Mesh & Texture Viewer"),
            
            html.Div([
                # Control Panel
                html.Div([
                    # File Loading Section
                    html.Div([
                        html.H3("📂 File Loading"),
                        html.Label("Mesh Directory:"),
                        dcc.Input(
                            id="mesh-directory",
                            type="text",
                            placeholder="/path/to/mesh/files",
                            value="/scratch/hdienye/marsfet_full_info/inflated_mesh",
                            style={'width': '100%'}
                        ),
                        html.Br(), html.Br(),
                        html.Label("Texture Directory:"),
                        dcc.Input(
                            id="texture-directory",
                            type="text",
                            placeholder="/path/to/texture/files",
                            value="/scratch/hdienye/marsfet_full_info/spangy/textures",
                            style={'width': '100%'}
                        ),
                        html.Br(), html.Br(),
                        html.Button("🔍 Scan Directories", id="scan-btn", className="btn btn-success", style={'width': '100%'}),
                        html.Div(id="file-status", style={'margin-top': '10px'}),
                    ], className="section"),
                    
                    # File Selection Section
                    html.Div([
                        html.H3("🎯 File Selection"),
                        html.Label("Available Mesh Files:"),
                        dcc.Dropdown(
                            id="mesh-dropdown",
                            placeholder="Select a mesh file...",
                            style={'margin-bottom': '15px'}
                        ),
                        html.Label("Available Texture Files:"),
                        dcc.Dropdown(
                            id="texture-dropdown",
                            placeholder="Select a texture file...",
                            style={'margin-bottom': '15px'}
                        ),
                        html.Button("📊 Load Selected Files", id="load-btn", className="btn btn-success", style={'width': '100%'}),
                    ], className="section"),
                    
                    # Display Options Section
                    html.Div([
                        html.H3("👁️ Display Options"),
                        html.Div([
                            html.Button("Mesh Only", id="mesh-view-btn", className="btn btn-info", style={'margin-right': '10px'}),
                            html.Button("With Texture", id="texture-view-btn", className="btn btn-warning"),
                        ], style={'margin-bottom': '15px'}),
                        
                        html.Label("Hemisphere:"),
                        dcc.Dropdown(
                            id="hemisphere-dropdown",
                            options=[
                                {"label": "Left", "value": "left"},
                                {"label": "Right", "value": "right"}
                            ],
                            value="left",
                            style={'margin-bottom': '15px'}
                        ),
                        
                        html.Label("View Type:"),
                        dcc.Dropdown(
                            id="view-type-dropdown",
                            options=[
                                {"label": "Both (Gyri + Sulci)", "value": "both"},
                                {"label": "Positive Only (Gyri)", "value": "positive"},
                                {"label": "Negative Only (Sulci)", "value": "negative"}
                            ],
                            value="both",
                            style={'margin-bottom': '15px'}
                        ),
                    ], className="section"),
                    
                    # Band Selection Section
                    html.Div([
                        html.H3("🎛️ Band Selection"),
                        dcc.Checklist(
                            id="band-checklist",
                            options=[
                                {"label": " B-6 (Sulci)", "value": -6},
                                {"label": " B-5 (Sulci)", "value": -5},
                                {"label": " B-4 (Sulci)", "value": -4},
                                {"label": " B1 (Gyri)", "value": 1},
                                {"label": " B2 (Gyri)", "value": 2},
                                {"label": " B3 (Gyri)", "value": 3},
                                {"label": " B4 (Gyri)", "value": 4},
                                {"label": " B5 (Gyri)", "value": 5},
                                {"label": " B6 (Gyri)", "value": 6},
                            ],
                            value=[-6, -5, -4, 1, 2, 3, 4, 5, 6],
                            inline=True,
                            style={'margin': '10px 0'}
                        ),
                    ], className="section"),
                    
                    # Camera & Rendering Section
                    html.Div([
                        html.H3("📹 Camera & Rendering"),
                        html.Div([
                            html.Button("Medial View", id="medial-btn", className="btn btn-secondary", style={'margin-right': '10px'}),
                            html.Button("Lateral View", id="lateral-btn", className="btn btn-secondary"),
                        ], style={'margin-bottom': '15px'}),
                        
                        html.Label("Opacity:"),
                        dcc.Slider(
                            id="opacity-slider",
                            min=0.1, max=1.0, step=0.1, value=1.0,
                            marks={i/10: f'{i/10}' for i in range(1, 11)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Br(),
                        
                        html.Label("Texture Intensity:"),
                        dcc.Slider(
                            id="intensity-slider",
                            min=0.1, max=2.0, step=0.1, value=1.0,
                            marks={i/10: f'{i/10}' for i in range(1, 21, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], className="section"),
                    
                    # Quality Control Section
                    html.Div([
                        html.H3("🔍 Quality Control"),
                        html.Div([
                            html.Button("📊 Mesh Info", id="mesh-info-btn", className="btn btn-info", style={'margin': '4px'}),
                            html.Button("🎨 Texture Stats", id="texture-stats-btn", className="btn btn-warning", style={'margin': '4px'}),
                            html.Button("✅ Validate", id="validate-btn", className="btn btn-success", style={'margin': '4px'}),
                            html.Button("💾 Export", id="export-btn", className="btn btn-danger", style={'margin': '4px'}),
                        ]),
                        html.Div(id="quality-info", style={'margin-top': '15px'})
                    ], className="section"),
                    
                ], className="control-panel"),
                
                # Visualization Panel
                html.Div([
                    dcc.Graph(
                        id="mesh-plot",
                        style={'height': '85vh'},
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'mesh_view',
                                'height': 1200,
                                'width': 1200,
                                'scale': 2
                            }
                        }
                    ),
                    html.Div(id="plot-info", style={'text-align': 'center', 'margin-top': '10px', 'color': '#d1d5db'})
                ], className="visualization-panel")
            ], className="main-container"),
            
            # Hidden divs for storing data
            html.Div(id="current-mesh-data", style={"display": "none"}),
            html.Div(id="current-texture-data", style={"display": "none"}),
            html.Div(id="current-view-mode", children="mesh", style={"display": "none"}),
        ])
    
    def setup_callbacks(self):
        """Set up all the callback functions"""
        
        @self.app.callback(
            [Output("mesh-dropdown", "options"),
             Output("texture-dropdown", "options"),
             Output("file-status", "children")],
            [Input("scan-btn", "n_clicks")],
            [State("mesh-directory", "value"),
             State("texture-directory", "value")]
        )
        def scan_directories(n_clicks, mesh_dir, texture_dir):
            if not n_clicks:
                return [], [], ""
            
            mesh_options = []
            texture_options = []
            status_messages = []
            
            # Scan mesh directory
            if mesh_dir and os.path.exists(mesh_dir):
                mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith(('.gii', '.surf.gii'))]
                mesh_options = [{"label": f, "value": os.path.join(mesh_dir, f)} for f in mesh_files]
                status_messages.append(f"✅ Found {len(mesh_files)} mesh files")
            else:
                status_messages.append("❌ Mesh directory not found")
            
            # Scan texture directory
            if texture_dir and os.path.exists(texture_dir):
                texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.gii')]
                texture_options = [{"label": f, "value": os.path.join(texture_dir, f)} for f in texture_files]
                status_messages.append(f"✅ Found {len(texture_files)} texture files")
            else:
                status_messages.append("❌ Texture directory not found")
            
            status_div = html.Div([
                html.P(msg, style={'margin': '5px 0', 'color': '#34d399' if '✅' in msg else '#f87171'}) 
                for msg in status_messages
            ], className="info-panel" if all('✅' in msg for msg in status_messages) else "error-panel")
            
            return mesh_options, texture_options, status_div
        
        @self.app.callback(
            [Output("current-mesh-data", "children"),
             Output("current-texture-data", "children"),
             Output("plot-info", "children")],
            [Input("load-btn", "n_clicks")],
            [State("mesh-dropdown", "value"),
             State("texture-dropdown", "value"),
             State("hemisphere-dropdown", "value")]
        )
        def load_files(n_clicks, mesh_path, texture_path, hemisphere):
            if not n_clicks or not mesh_path:
                return "", "", "No files loaded"
            
            try:
                # Load mesh
                mesh = sio.load_mesh(mesh_path)
                mesh, camera_medial, camera_lateral = self.mesh_orientation(mesh, hemisphere)
                
                mesh_data = {
                    "vertices": mesh.vertices.tolist(),
                    "faces": mesh.faces.tolist(),
                    "camera_medial": camera_medial,
                    "camera_lateral": camera_lateral
                }
                
                info_parts = [f"📐 Mesh: {os.path.basename(mesh_path)} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)"]
                
                # Load texture if provided
                texture_data = ""
                if texture_path:
                    scalars = self.read_gii_file(texture_path)
                    if scalars is not None:
                        texture_data = json.dumps({"scalars": scalars.tolist()})
                        info_parts.append(f"🎨 Texture: {os.path.basename(texture_path)} ({len(scalars)} values)")
                
                return json.dumps(mesh_data), texture_data, " | ".join(info_parts)
                
            except Exception as e:
                return "", "", f"❌ Error loading files: {str(e)}"
        
        @self.app.callback(
            Output("current-view-mode", "children"),
            [Input("mesh-view-btn", "n_clicks"),
             Input("texture-view-btn", "n_clicks")]
        )
        def update_view_mode(mesh_clicks, texture_clicks):
            ctx = callback_context
            if not ctx.triggered:
                return "mesh"
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            return "mesh" if button_id == "mesh-view-btn" else "texture"
        
        @self.app.callback(
            Output("mesh-plot", "figure"),
            [Input("current-mesh-data", "children"),
             Input("current-texture-data", "children"),
             Input("current-view-mode", "children"),
             Input("hemisphere-dropdown", "value"),
             Input("view-type-dropdown", "value"),
             Input("band-checklist", "value"),
             Input("opacity-slider", "value"),
             Input("intensity-slider", "value"),
             Input("medial-btn", "n_clicks"),
             Input("lateral-btn", "n_clicks")]
        )
        def update_plot(mesh_data, texture_data, view_mode, hemisphere, view_type, 
                       selected_bands, opacity, intensity, medial_clicks, lateral_clicks):
            
            if not mesh_data:
                fig = go.Figure()
                fig.add_annotation(
                    text="🔄 Load mesh and texture files to begin visualization",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20, color="white")
                )
                fig.update_layout(
                    paper_bgcolor='rgb(26, 26, 46)',
                    plot_bgcolor='rgb(26, 26, 46)',
                    font=dict(color='white')
                )
                return fig
            
            # Parse mesh data
            mesh_info = json.loads(mesh_data)
            vertices = np.array(mesh_info["vertices"])
            faces = np.array(mesh_info["faces"])
            
            # Determine camera position
            ctx = callback_context
            camera = mesh_info["camera_lateral"]  # Default
            if ctx.triggered:
                button_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if button_id == "medial-btn":
                    camera = mesh_info["camera_medial"]
                elif button_id == "lateral-btn":
                    camera = mesh_info["camera_lateral"]
            
            # Create base mesh
            fig = go.Figure()
            
            if view_mode == "mesh" or not texture_data:
                # Mesh only view
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color='lightgray',
                    opacity=opacity,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2),
                    showscale=False,
                    hovertemplate='Vertex: %{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
                ))
            else:
                # Textured view
                texture_info = json.loads(texture_data)
                scalars = np.array(texture_info["scalars"])
                
                # Apply filtering
                display_scalars = scalars.copy()
                
                # Band selection
                if selected_bands:
                    mask = np.zeros_like(display_scalars, dtype=bool)
                    for band in selected_bands:
                        mask |= (np.round(display_scalars) == band)
                    display_scalars[~mask] = np.nan
                
                # View type filtering
                if view_type == 'positive':
                    display_scalars[display_scalars < 0] = np.nan
                elif view_type == 'negative':
                    display_scalars[display_scalars > 0] = np.nan
                
                # Apply intensity and clipping
                display_scalars = np.clip(display_scalars * intensity, -6, 6)
                
                # Create colormap
                colorscale, value_color_map = self.create_custom_colormap()
                
                fig.add_trace(go.Mesh3d(
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
                    opacity=opacity,
                    showscale=False,
                    hovertemplate='Value: %{intensity:.2f}<br>Position: %{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>'
                ))
                
                # Add legend traces
                if selected_bands:
                    legend_bands = [b for b in selected_bands if 
                                   (b < 0 and view_type in ['both', 'negative']) or
                                   (b > 0 and view_type in ['both', 'positive'])]
                    
                    for band in sorted(legend_bands):
                        if band in value_color_map:
                            fig.add_trace(go.Scatter3d(
                                x=[None], y=[None], z=[None],
                                mode='markers',
                                marker=dict(size=10, color=value_color_map[band]),
                                name=f'B{band}',
                                showlegend=True
                            ))
            
            # Update layout
            show_legend = bool(view_mode == "texture" and texture_data)
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=camera,
                    aspectmode='cube',
                    bgcolor='rgb(26, 26, 46)'
                ),
                paper_bgcolor='rgb(26, 26, 46)',
                plot_bgcolor='rgb(26, 26, 46)',
                font=dict(color='white'),
                showlegend=show_legend,
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="right", x=0.99,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    font=dict(color='black')
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            
            return fig
        
        @self.app.callback(
            Output("quality-info", "children"),
            [Input("mesh-info-btn", "n_clicks"),
             Input("texture-stats-btn", "n_clicks"),
             Input("validate-btn", "n_clicks")],
            [State("current-mesh-data", "children"),
             State("current-texture-data", "children")]
        )
        def quality_control(mesh_info_clicks, texture_stats_clicks, validate_clicks,
                          mesh_data, texture_data):
            ctx = callback_context
            if not ctx.triggered or not mesh_data:
                return ""
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            mesh_info = json.loads(mesh_data)
            vertices = np.array(mesh_info["vertices"])
            faces = np.array(mesh_info["faces"])
            
            if button_id == "mesh-info-btn":
                stats = {
                    'vertices': len(vertices),
                    'faces': len(faces),
                    'x_range': f"{vertices[:, 0].min():.2f} to {vertices[:, 0].max():.2f}",
                    'y_range': f"{vertices[:, 1].min():.2f} to {vertices[:, 1].max():.2f}",
                    'z_range': f"{vertices[:, 2].min():.2f} to {vertices[:, 2].max():.2f}"
                }
                return html.Div([
                    html.H4("📊 Mesh Information", style={'color': '#64b5f6'}),
                    html.P(f"Vertices: {stats['vertices']}"),
                    html.P(f"Faces: {stats['faces']}"),
                    html.P(f"X Range: {stats['x_range']}"),
                    html.P(f"Y Range: {stats['y_range']}"),
                    html.P(f"Z Range: {stats['z_range']}")
                ], className="info-panel")
                
            elif button_id == "texture-stats-btn" and texture_data:
                texture_info = json.loads(texture_data)
                scalars = np.array(texture_info["scalars"])
                valid_scalars = scalars[~np.isnan(scalars)]
                
                # Band distribution
                band_info = []
                for i in range(-6, 7):
                    if i != 0:
                        count = np.sum(np.round(valid_scalars) == i)
                        percentage = count/len(valid_scalars)*100 if len(valid_scalars) > 0 else 0
                        band_info.append(html.P(f"B{i}: {count} ({percentage:.1f}%)"))
                
                return html.Div([
                    html.H4("🎨 Texture Statistics", style={'color': '#64b5f6'}),
                    html.P(f"Total values: {len(scalars)}"),
                    html.P(f"Valid values: {len(valid_scalars)}"),
                    html.P(f"Range: {valid_scalars.min():.2f} to {valid_scalars.max():.2f}"),
                    html.P(f"Mean: {valid_scalars.mean():.2f}"),
                    html.Hr(style={'border-color': '#4b5563'}),
                    html.H5("Band Distribution:", style={'color': '#60a5fa'}),
                    html.Div(band_info)
                ], className="info-panel")
                
            elif button_id == "validate-btn":
                issues = []
                
                # Check for degenerate triangles
                degenerate_count = 0
                for face in faces:
                    if len(set(face)) < 3:
                        degenerate_count += 1
                
                if degenerate_count > 0:
                    issues.append(f"⚠️ {degenerate_count} degenerate triangles found")
                
                # Check vertex indices
                max_vertex_idx = len(vertices) - 1
                invalid_indices = np.sum((faces < 0) | (faces > max_vertex_idx))
                if invalid_indices > 0:
                    issues.append(f"⚠️ {invalid_indices} invalid vertex indices")
                
                # Check vertex-texture mismatch
                if texture_data:
                    texture_info = json.loads(texture_data)
                    scalars = np.array(texture_info["scalars"])
                    if len(vertices) != len(scalars):
                        issues.append(f"⚠️ Vertex-texture count mismatch: {len(vertices)} vs {len(scalars)}")
                
                if issues:
                    content = [html.P(issue, style={'color': '#f87171'}) for issue in issues]
                    panel_class = "error-panel"
                    title_color = '#f87171'
                else:
                    content = [html.P("✅ All validation checks passed!", style={'color': '#34d399'})]
                    panel_class = "success-panel"
                    title_color = '#34d399'
                
                return html.Div([
                    html.H4("✅ Validation Results", style={'color': title_color}),
                    html.Div(content)
                ], className=panel_class)
            
            return ""
    
    def run(self, debug=True, port=8050, host='127.0.0.1'):
        """Run the Dash app"""
        print(f"""
🧠 Interactive Mesh & Texture Viewer
====================================

🚀 Starting server at: http://{host}:{port}
📂 Default directories configured for your neuroimaging workflow
🎯 Ready for real-time quality control and visualization!

Features:
- Load GIFTI mesh (.gii, .surf.gii) and texture (.gii) files
- Real-time band selection and filtering (B-6 to B6)
- Hemisphere-specific orientations (left/right)
- Quality control tools (validation, statistics, export)
- Interactive 3D visualization with camera controls

Usage:
1. Update directory paths in the UI if needed
2. Click 'Scan Directories' to find available files
3. Select mesh and texture files from dropdowns
4. Click 'Load Selected Files' to visualize
5. Use controls for real-time manipulation
6. Quality control tools for validation and analysis

Press Ctrl+C to stop the server.
        """)
        
        self.app.run(debug=debug, port=port, host=host)

# Example usage and main execution
if __name__ == "__main__":
    # Create and run the viewer
    viewer = InteractiveMeshViewer()
    
    # Run the server - no additional dependencies needed!
    viewer.run(debug=True, port=8050)