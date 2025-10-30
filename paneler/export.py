"""Export 2D cutting patterns using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
import trimesh

from . import flatten


def plot_panel_pattern(
    vertices_2d: np.ndarray,
    panel_info: dict,
    title: str = "Panel Pattern",
    show_measurements: bool = True,
    seam_allowance: float = 0,
    curved_edges: list = None
) -> plt.Figure:
    """
    Plot a single panel pattern.

    Args:
        vertices_2d: 2D vertices of the panel
        panel_info: Panel information dict from flatten.get_panel_info()
        title: Panel title
        show_measurements: Whether to show edge measurements
        seam_allowance: Seam allowance distance (if already added to vertices)
        curved_edges: List of 2D curve arrays for each edge (if None, draws straight edges)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the main panel
    if curved_edges is not None:
        # Draw curved edges
        for edge_curve in curved_edges:
            ax.plot(edge_curve[:, 0], edge_curve[:, 1], 'k-', linewidth=2)
    else:
        # Draw straight edges (old behavior)
        poly = Polygon(vertices_2d, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(poly)

    # Draw seam allowance if specified
    if seam_allowance > 0:
        vertices_with_seam = flatten.add_seam_allowance(vertices_2d, seam_allowance)
        poly_seam = Polygon(vertices_with_seam, fill=False, edgecolor='red',
                          linewidth=1, linestyle='--', label='Seam allowance')
        ax.add_patch(poly_seam)

    # Add measurements
    if show_measurements:
        edge_lengths = panel_info['edge_lengths_2d']
        n = len(vertices_2d)

        for i in range(n):
            p0 = vertices_2d[i]
            p1 = vertices_2d[(i + 1) % n]

            # Midpoint
            mid = (p0 + p1) / 2

            # Perpendicular offset for text
            edge_vec = p1 - p0
            perp = np.array([-edge_vec[1], edge_vec[0]])
            perp = perp / np.linalg.norm(perp) * 0.1

            # Add text
            length = edge_lengths[i]
            ax.text(mid[0] + perp[0], mid[1] + perp[1], f'{length:.3f}',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add title with panel info
    info_text = f"{title}\n"
    info_text += f"Sides: {panel_info['n_sides']}, "
    info_text += f"Area: {panel_info['area_2d']:.4f}, "
    info_text += f"Perimeter: {panel_info['perimeter_2d']:.4f}"

    ax.set_title(info_text)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

    # Set limits with padding
    all_points = vertices_2d
    if seam_allowance > 0:
        all_points = np.vstack([all_points, vertices_with_seam])

    margin = 0.2
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    if seam_allowance > 0:
        ax.legend()

    plt.tight_layout()
    return fig


def export_all_panels(
    mesh,
    face_groups: dict,
    output_file: str = 'panels.pdf',
    seam_allowance: float = 0,
    flatten_method: str = 'stereographic'
):
    """
    Export all unique panel patterns to a PDF file.

    Args:
        mesh: Spherical mesh (PolyhedronMesh or trimesh.Trimesh)
        face_groups: Dictionary mapping group_id -> list of face indices
        output_file: Output PDF filename
        seam_allowance: Seam allowance to add (in same units as mesh)
        flatten_method: Method for flattening ('stereographic' or 'azimuthal')
    """
    from . import geometry

    # Get faces list
    if isinstance(mesh, geometry.PolyhedronMesh):
        faces = mesh.faces
    else:
        faces = mesh.faces.tolist()

    with PdfPages(output_file) as pdf:
        # Summary page
        fig_summary = _create_summary_page(mesh, face_groups)
        pdf.savefig(fig_summary)
        plt.close(fig_summary)

        # Create one page per unique panel type
        for group_id, face_indices in face_groups.items():
            # Take the first face as representative
            face_idx = face_indices[0]
            face = faces[face_idx]
            vertices_3d = mesh.vertices[face]

            # Flatten to 2D
            vertices_2d = flatten.flatten_spherical_face(vertices_3d, method=flatten_method)

            # Get curved edges with vertices_2d to ensure proper connection
            curved_edges = flatten.get_curved_edges_2d(vertices_3d, vertices_2d, method=flatten_method, n_points=30)

            # Get panel info
            panel_info = flatten.get_panel_info(vertices_3d, vertices_2d)

            # Create plot
            title = f"Panel Type {group_id + 1} (Quantity: {len(face_indices)})"
            fig = plot_panel_pattern(
                vertices_2d,
                panel_info,
                title=title,
                show_measurements=True,
                seam_allowance=seam_allowance,
                curved_edges=curved_edges
            )

            pdf.savefig(fig)
            plt.close(fig)

    print(f"Exported {len(face_groups)} panel types to {output_file}")


def _create_summary_page(mesh, face_groups: dict) -> plt.Figure:
    """Create a summary page with panel statistics."""
    from . import geometry

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')

    # Title
    title_text = "Panel Cutting Patterns - Summary\n\n"
    ax.text(0.5, 0.95, title_text, ha='center', va='top',
           fontsize=16, fontweight='bold', transform=ax.transAxes)

    # Get faces list
    if isinstance(mesh, geometry.PolyhedronMesh):
        faces = mesh.faces
        total_faces = len(faces)
        total_vertices = len(mesh.vertices)
        total_edges = len(mesh.trimesh.edges_unique)
    else:
        faces = mesh.faces.tolist()
        total_faces = len(mesh.faces)
        total_vertices = len(mesh.vertices)
        total_edges = len(mesh.edges_unique)

    n_panel_types = len(face_groups)

    stats_text = "Mesh Statistics:\n"
    stats_text += f"  Total Faces: {total_faces}\n"
    stats_text += f"  Total Vertices: {total_vertices}\n"
    stats_text += f"  Total Edges: {total_edges}\n"
    stats_text += f"  Unique Panel Types: {n_panel_types}\n\n"

    ax.text(0.1, 0.85, stats_text, ha='left', va='top',
           fontsize=12, family='monospace', transform=ax.transAxes)

    # Panel breakdown
    panel_text = "Panel Breakdown:\n\n"

    for group_id, face_indices in sorted(face_groups.items()):
        # Get representative face
        face_idx = face_indices[0]
        face = faces[face_idx]
        n_sides = len(face)

        panel_text += f"  Panel Type {group_id + 1}:\n"
        panel_text += f"    Quantity: {len(face_indices)}\n"
        panel_text += f"    Sides: {n_sides}\n\n"

    ax.text(0.1, 0.65, panel_text, ha='left', va='top',
           fontsize=11, family='monospace', transform=ax.transAxes)

    # Instructions
    instructions = "\nInstructions:\n\n"
    instructions += "1. Print each panel pattern on the following pages\n"
    instructions += "2. Cut along the solid black lines\n"
    instructions += "3. Use the dashed red lines for seam allowance (if shown)\n"
    instructions += "4. Edge measurements are shown in the same units as input\n"
    instructions += "5. Assemble panels according to the 3D model\n"

    ax.text(0.1, 0.35, instructions, ha='left', va='top',
           fontsize=10, transform=ax.transAxes)

    return fig


def export_single_panel(
    vertices_3d: np.ndarray,
    output_file: str = 'panel.png',
    seam_allowance: float = 0,
    flatten_method: str = 'stereographic',
    dpi: int = 300
):
    """
    Export a single panel pattern to an image file.

    Args:
        vertices_3d: 3D vertices on sphere
        output_file: Output filename (.png, .pdf, .svg, etc.)
        seam_allowance: Seam allowance to add
        flatten_method: Flattening method
        dpi: Resolution for raster formats
    """
    # Flatten to 2D
    vertices_2d = flatten.flatten_spherical_face(vertices_3d, method=flatten_method)

    # Get curved edges
    curved_edges = flatten.get_curved_edges_2d(vertices_3d, vertices_2d, method=flatten_method, n_points=30)

    # Get panel info
    panel_info = flatten.get_panel_info(vertices_3d, vertices_2d)

    # Create plot
    fig = plot_panel_pattern(
        vertices_2d,
        panel_info,
        title="Panel Pattern",
        show_measurements=True,
        seam_allowance=seam_allowance,
        curved_edges=curved_edges
    )

    # Save
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Exported panel pattern to {output_file}")


def create_layout_sheet(
    mesh,
    face_groups: dict,
    sheet_width: float = 1.0,
    sheet_height: float = 1.0,
    output_file: str = 'layout.pdf',
    seam_allowance: float = 0,
    flatten_method: str = 'stereographic'
):
    """
    Create a layout sheet with all panels arranged for cutting.

    This is a simple version that places panels in a grid.
    A more sophisticated version would use nesting algorithms.

    Args:
        mesh: Spherical mesh (PolyhedronMesh or trimesh.Trimesh)
        face_groups: Face groups
        sheet_width: Width of the cutting sheet
        sheet_height: Height of the cutting sheet
        output_file: Output filename
        seam_allowance: Seam allowance
        flatten_method: Flattening method
    """
    from . import geometry

    # Get faces list
    if isinstance(mesh, geometry.PolyhedronMesh):
        faces = mesh.faces
    else:
        faces = mesh.faces.tolist()

    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Draw sheet boundary
    sheet_rect = plt.Rectangle((0, 0), sheet_width, sheet_height,
                              fill=False, edgecolor='blue',
                              linewidth=2, label='Sheet boundary')
    ax.add_patch(sheet_rect)

    # Simple grid layout
    x_offset = 0.05
    y_offset = 0.05
    spacing = 0.05

    for group_id, face_indices in face_groups.items():
        # Process each instance of this panel type
        for face_idx in face_indices:
            face = faces[face_idx]
            vertices_3d = mesh.vertices[face]

            # Flatten
            vertices_2d = flatten.flatten_spherical_face(vertices_3d, method=flatten_method)

            # Get curved edges
            curved_edges = flatten.get_curved_edges_2d(vertices_3d, vertices_2d, method=flatten_method, n_points=20)

            # Add seam allowance if specified
            if seam_allowance > 0:
                vertices_2d = flatten.add_seam_allowance(vertices_2d, seam_allowance)

            # Center at origin
            center = np.mean(vertices_2d, axis=0)
            vertices_2d = vertices_2d - center

            # Scale to fit
            max_extent = np.max(np.abs(vertices_2d)) * 2.5

            # Check if we need to move to next row
            if x_offset + max_extent > sheet_width - 0.05:
                x_offset = 0.05
                y_offset += max_extent + spacing

            if y_offset + max_extent > sheet_height:
                print("Warning: Not all panels fit on sheet")
                break

            # Translate curved edges to position
            offset = np.array([x_offset + max_extent/2, y_offset + max_extent/2]) - center

            # Draw curved edges
            for edge_curve in curved_edges:
                positioned_curve = edge_curve + offset
                ax.plot(positioned_curve[:, 0], positioned_curve[:, 1], 'k-', linewidth=1)

            # Move to next position
            x_offset += max_extent + spacing

    ax.set_xlim(-0.02, sheet_width + 0.02)
    ax.set_ylim(-0.02, sheet_height + 0.02)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Panel Layout Sheet')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Exported layout sheet to {output_file}")
