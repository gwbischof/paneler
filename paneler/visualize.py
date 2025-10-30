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
    Visualize spherical projection with original polyhedron inside.

    Args:
        original_mesh: Original polyhedron (PolyhedronMesh or trimesh.Trimesh)
        spherical_mesh: Spherically projected mesh (PolyhedronMesh or trimesh.Trimesh)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # Original polyhedron - scale to 0.6 to show it's inside the sphere
    tm_orig = _get_trimesh(original_mesh)
    v_orig = tm_orig.vertices * 0.6  # Scale down to show difference
    f_orig = tm_orig.faces

    # Add original polyhedron mesh
    fig.add_trace(
        go.Mesh3d(
            x=v_orig[:, 0], y=v_orig[:, 1], z=v_orig[:, 2],
            i=f_orig[:, 0], j=f_orig[:, 1], k=f_orig[:, 2],
            color='lightblue',
            opacity=0.7,
            flatshading=True,
            name='Original',
        )
    )

    # Add edges to original polyhedron (from original polygon faces, not triangulated)
    from . import geometry
    orig_edge_x = []
    orig_edge_y = []
    orig_edge_z = []

    if isinstance(original_mesh, geometry.PolyhedronMesh):
        # Use original polygon faces to get edges (no diagonals)
        for face in original_mesh.faces:
            for i in range(len(face)):
                v0 = face[i]
                v1 = face[(i + 1) % len(face)]
                p0 = v_orig[v0]
                p1 = v_orig[v1]
                orig_edge_x.extend([p0[0], p1[0], None])
                orig_edge_y.extend([p0[1], p1[1], None])
                orig_edge_z.extend([p0[2], p1[2], None])
    else:
        # Fallback to triangulated edges
        for edge in tm_orig.edges_unique:
            v0, v1 = edge
            p0 = v_orig[v0]
            p1 = v_orig[v1]
            orig_edge_x.extend([p0[0], p1[0], None])
            orig_edge_y.extend([p0[1], p1[1], None])
            orig_edge_z.extend([p0[2], p1[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=orig_edge_x, y=orig_edge_y, z=orig_edge_z,
            mode='lines',
            line=dict(color='darkblue', width=2),
            hoverinfo='skip',
            showlegend=False,
        )
    )

    # Spherical projection - just the edges, no mesh
    tm_sphere = _get_trimesh(spherical_mesh)
    v_sphere = tm_sphere.vertices
    f_sphere = tm_sphere.faces

    # Add geodesic edges to spherical projection (from original polygon edges)
    edge_x = []
    edge_y = []
    edge_z = []

    # Get edges from original faces to avoid diagonals
    edge_pairs = set()
    if isinstance(spherical_mesh, geometry.PolyhedronMesh):
        for face in spherical_mesh.faces:
            for i in range(len(face)):
                v0 = face[i]
                v1 = face[(i + 1) % len(face)]
                # Store as sorted tuple to avoid duplicates
                edge_pairs.add(tuple(sorted([v0, v1])))
    else:
        for edge in tm_sphere.edges_unique:
            edge_pairs.add(tuple(sorted(edge)))

    for edge in edge_pairs:
        v0, v1 = edge
        p0 = v_sphere[v0]
        p1 = v_sphere[v1]

        # Create geodesic curve along sphere surface
        n_points = 20
        for i in range(n_points):
            t = i / (n_points - 1)
            # Spherical linear interpolation (slerp)
            cos_angle = np.dot(p0, p1)
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            if angle < 0.001:  # Points are very close
                p = (1 - t) * p0 + t * p1
            else:
                # Slerp formula
                sin_angle = np.sin(angle)
                p = (np.sin((1 - t) * angle) / sin_angle) * p0 + (np.sin(t * angle) / sin_angle) * p1

            # Normalize to ensure point is on sphere
            p = p / np.linalg.norm(p)

            edge_x.append(p[0])
            edge_y.append(p[1])
            edge_z.append(p[2])

        # Add None to separate line segments
        edge_x.append(None)
        edge_y.append(None)
        edge_z.append(None)

    fig.add_trace(
        go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='darkred', width=3),
            hoverinfo='skip',
            showlegend=False,
        )
    )

    # Add reference sphere
    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 20)
    theta, phi = np.meshgrid(theta, phi)

    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)

    fig.add_trace(
        go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'rgba(255,150,150,0.1)'], [1, 'rgba(255,150,150,0.1)']],
            showscale=False,
            hoverinfo='skip',
            name='Sphere',
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(text='Spherical Projection', x=0.5, xanchor='center'),
        showlegend=False,
        autosize=True,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube',
            camera=dict(
                center=dict(x=0, y=0, z=0)
            ),
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
