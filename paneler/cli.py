"""Command-line interface for paneler."""

import argparse
from . import geometry, visualize, export


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate soccer ball and footbag panel cutting patterns'
    )

    parser.add_argument(
        'polyhedron',
        type=str,
        nargs='?',
        help='Polyhedron type (tetrahedron, cube, octahedron, dodecahedron, '
             'icosahedron, truncated_icosahedron)'
    )

    parser.add_argument(
        '--radius',
        type=float,
        default=1.0,
        help='Sphere radius for projection (default: 1.0)'
    )

    parser.add_argument(
        '--seam-allowance',
        type=float,
        default=0.0,
        help='Seam allowance to add to panels (default: 0.0)'
    )

    parser.add_argument(
        '--flatten-method',
        type=str,
        default='stereographic',
        choices=['stereographic', 'azimuthal'],
        help='Method for flattening spherical panels (default: stereographic)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='panels.pdf',
        help='Output PDF file for cutting patterns (default: panels.pdf)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show 3D visualization in browser'
    )

    parser.add_argument(
        '--save-viz',
        type=str,
        help='Save 3D visualization to HTML file'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available polyhedra and exit'
    )

    args = parser.parse_args()

    # List available polyhedra
    if args.list:
        print("Available polyhedra:")
        for name in geometry.list_available_polyhedra():
            print(f"  - {name}")
        return

    # Check if polyhedron is provided
    if not args.polyhedron:
        parser.error("the following arguments are required: polyhedron")

    # Generate polyhedron
    print(f"Generating {args.polyhedron}...")
    try:
        mesh = geometry.get_polyhedron(args.polyhedron)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable polyhedra:")
        for name in geometry.list_available_polyhedra():
            print(f"  - {name}")
        return

    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")

    # Project to sphere
    print(f"\nProjecting to sphere (radius={args.radius})...")
    spherical_mesh = geometry.project_to_sphere(mesh, radius=args.radius)

    # Identify unique panel types
    print("\nIdentifying unique panel types...")
    face_groups = geometry.get_face_groups(spherical_mesh)
    print(f"  Found {len(face_groups)} unique panel types:")

    for group_id, face_indices in face_groups.items():
        face = spherical_mesh.faces[face_indices[0]]
        n_sides = len(face)
        print(f"    Type {group_id + 1}: {n_sides}-sided, quantity {len(face_indices)}")

    # Export cutting patterns
    print(f"\nExporting cutting patterns to {args.output}...")
    export.export_all_panels(
        spherical_mesh,
        face_groups,
        output_file=args.output,
        seam_allowance=args.seam_allowance,
        flatten_method=args.flatten_method
    )

    # Visualization
    if args.visualize or args.save_viz:
        print("\nCreating 3D visualization...")
        fig = visualize.visualize_with_face_colors(
            spherical_mesh,
            face_groups,
            title=f"{args.polyhedron.replace('_', ' ').title()} - Spherical Projection"
        )

        if args.save_viz:
            visualize.save_figure(fig, args.save_viz)

        if args.visualize:
            visualize.show_figure(fig)

    print("\nDone!")


if __name__ == '__main__':
    main()
