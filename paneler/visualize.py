"""3D visualization using plotly."""

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots


def _get_trimesh(mesh):
    """Helper to get trimesh from either PolyhedronMesh or trimesh.Trimesh."""
    from . import geometry
    if isinstance(mesh, geometry.PolyhedronMesh):
        return mesh.trimesh
    return mesh


def visualize_polyhedron(mesh, title: str = "Polyhedron") -> go.Figure:
    """
    Visualize a polyhedron using plotly.

    Args:
        mesh: The mesh to visualize (PolyhedronMesh or trimesh.Trimesh)
        title: Plot title

    Returns:
        Plotly figure
    """
    tm = _get_trimesh(mesh)
    vertices = tm.vertices
    faces = tm.faces

    # Create mesh plot
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=0.8,
            flatshading=True,
        )
    ])

    # Add edges
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in tm.edges_unique:
        v0, v1 = edge
        p0 = vertices[v0]
        p1 = vertices[v1]

        edge_x.extend([p0[0], p1[0], None])
        edge_y.extend([p0[1], p1[1], None])
        edge_z.extend([p0[2], p1[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=2),
        name='Edges',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
        ),
        showlegend=False,
    )

    return fig


def visualize_with_face_colors(
    mesh,
    face_groups: dict,
    title: str = "Polyhedron with Colored Panels"
) -> go.Figure:
    """
    Visualize polyhedron with different colors for each face group.

    Args:
        mesh: The mesh to visualize (PolyhedronMesh or trimesh.Trimesh)
        face_groups: Dictionary mapping group_id -> list of face indices
        title: Plot title

    Returns:
        Plotly figure
    """
    from . import geometry

    tm = _get_trimesh(mesh)
    vertices = tm.vertices

    # Define colors for different groups
    colors = [
        'lightblue', 'lightcoral', 'lightgreen', 'lightyellow',
        'lightpink', 'lightcyan', 'lavender', 'peachpuff',
        'palegreen', 'plum', 'khaki', 'thistle'
    ]

    fig = go.Figure()

    # For PolyhedronMesh, we need to map original face indices to triangulated faces
    if isinstance(mesh, geometry.PolyhedronMesh):
        # Build mapping from original face to triangulated faces
        face_to_tris = {}
        tri_idx = 0
        for orig_face_idx, face in enumerate(mesh.faces):
            n_tris = len(face) - 2  # Fan triangulation
            face_to_tris[orig_face_idx] = list(range(tri_idx, tri_idx + n_tris))
            tri_idx += n_tris

        # Add each group with its own color
        for group_id, face_indices in face_groups.items():
            color = colors[group_id % len(colors)]

            # Get triangulated faces for this group
            tri_indices = []
            for orig_idx in face_indices:
                tri_indices.extend(face_to_tris[orig_idx])

            faces = tm.faces[tri_indices]

            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.9,
                flatshading=True,
                name=f'Panel type {group_id + 1} ({len(face_indices)} panels)',
                hoverinfo='name',
            ))
    else:
        # Add each group with its own color
        for group_id, face_indices in face_groups.items():
            color = colors[group_id % len(colors)]
            faces = tm.faces[face_indices]

            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.9,
                flatshading=True,
                name=f'Panel type {group_id + 1} ({len(face_indices)} panels)',
                hoverinfo='name',
            ))

    # Add edges
    edge_x = []
    edge_y = []
    edge_z = []

    for edge in tm.edges_unique:
        v0, v1 = edge
        p0 = vertices[v0]
        p1 = vertices[v1]

        edge_x.extend([p0[0], p1[0], None])
        edge_y.extend([p0[1], p1[1], None])
        edge_z.extend([p0[2], p1[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=2),
        name='Edges',
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
        ),
    )

    return fig


def visualize_comparison(
    original_mesh,
    spherical_mesh
) -> go.Figure:
    """
    Side-by-side comparison of original polyhedron and spherical projection.

    Args:
        original_mesh: Original polyhedron (PolyhedronMesh or trimesh.Trimesh)
        spherical_mesh: Spherically projected mesh (PolyhedronMesh or trimesh.Trimesh)

    Returns:
        Plotly figure with subplots
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original Polyhedron', 'Spherical Projection'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # Original polyhedron
    tm_orig = _get_trimesh(original_mesh)
    v_orig = tm_orig.vertices
    f_orig = tm_orig.faces

    fig.add_trace(
        go.Mesh3d(
            x=v_orig[:, 0], y=v_orig[:, 1], z=v_orig[:, 2],
            i=f_orig[:, 0], j=f_orig[:, 1], k=f_orig[:, 2],
            color='lightblue',
            opacity=0.8,
            flatshading=True,
        ),
        row=1, col=1
    )

    # Spherical projection
    tm_sphere = _get_trimesh(spherical_mesh)
    v_sphere = tm_sphere.vertices
    f_sphere = tm_sphere.faces

    fig.add_trace(
        go.Mesh3d(
            x=v_sphere[:, 0], y=v_sphere[:, 1], z=v_sphere[:, 2],
            i=f_sphere[:, 0], j=f_sphere[:, 1], k=f_sphere[:, 2],
            color='lightcoral',
            opacity=0.8,
            flatshading=True,
        ),
        row=1, col=2
    )

    # Add reference sphere (wireframe)
    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 20)
    theta, phi = np.meshgrid(theta, phi)

    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)

    fig.add_trace(
        go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'rgba(200,200,200,0.1)'], [1, 'rgba(200,200,200,0.1)']],
            showscale=False,
            hoverinfo='skip',
        ),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
        ),
        scene2=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
        ),
    )

    return fig


def show_figure(fig: go.Figure, auto_open: bool = True):
    """
    Display a plotly figure.

    Args:
        fig: Plotly figure
        auto_open: Whether to open in browser automatically
    """
    fig.show(auto_open=auto_open)


def save_figure(fig: go.Figure, filename: str):
    """
    Save figure to HTML file.

    Args:
        fig: Plotly figure
        filename: Output filename (should end with .html)
    """
    fig.write_html(filename)
    print(f"Saved visualization to {filename}")
