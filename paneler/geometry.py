"""Polyhedra generation and spherical projection."""

import numpy as np
import trimesh


class PolyhedronMesh:
    """
    Wrapper for polyhedron meshes that can have non-triangular faces.

    Attributes:
        vertices: Vertex positions (N x 3)
        faces: List of faces, each face is a list of vertex indices
        trimesh: Triangulated trimesh.Trimesh for visualization
    """

    def __init__(self, vertices: np.ndarray, faces: list):
        self.vertices = np.array(vertices)
        self.faces = faces  # List of lists (polygon faces)

        # Create triangulated version for visualization
        triangulated_faces = []
        for poly in faces:
            # Simple fan triangulation
            for i in range(1, len(poly) - 1):
                triangulated_faces.append([poly[0], poly[i], poly[i + 1]])

        self.trimesh = trimesh.Trimesh(vertices=self.vertices, faces=triangulated_faces)


def get_polyhedron(name: str):
    """
    Get a polyhedron by name.

    Args:
        name: Name of the polyhedron. Options:
            - 'tetrahedron'
            - 'cube' or 'hexahedron'
            - 'octahedron'
            - 'dodecahedron'
            - 'icosahedron'
            - 'truncated_icosahedron' (classic soccer ball)

    Returns:
        PolyhedronMesh or trimesh.Trimesh: The polyhedron mesh
    """
    name = name.lower().replace('-', '_').replace(' ', '_')

    # Built-in Platonic solids (all triangular faces)
    if name == 'tetrahedron':
        mesh = trimesh.creation.tetrahedron()
        # Convert to PolyhedronMesh for consistency
        faces = [list(f) for f in mesh.faces]
        return PolyhedronMesh(mesh.vertices, faces)
    elif name in ('cube', 'hexahedron'):
        mesh = trimesh.creation.box()
        faces = [list(f) for f in mesh.faces]
        return PolyhedronMesh(mesh.vertices, faces)
    elif name == 'octahedron':
        mesh = trimesh.creation.octahedron()
        faces = [list(f) for f in mesh.faces]
        return PolyhedronMesh(mesh.vertices, faces)
    elif name == 'dodecahedron':
        mesh = trimesh.creation.icosahedron()
        faces = [list(f) for f in mesh.faces]
        return PolyhedronMesh(mesh.vertices, faces)
    elif name == 'icosahedron':
        mesh = trimesh.creation.icosahedron()
        faces = [list(f) for f in mesh.faces]
        return PolyhedronMesh(mesh.vertices, faces)
    elif name == 'truncated_icosahedron':
        # Classic soccer ball: 12 pentagons + 20 hexagons
        # Start with icosahedron and truncate vertices
        ico = trimesh.creation.icosahedron()
        # Truncate by a factor (0.33 is typical for soccer ball proportions)
        return _truncate_polyhedron(ico, truncation_factor=0.33)
    else:
        raise ValueError(f"Unknown polyhedron: {name}")


def _truncate_polyhedron(mesh: trimesh.Trimesh, truncation_factor: float = 0.33) -> PolyhedronMesh:
    """
    Truncate vertices of a polyhedron.

    This creates new faces by cutting off each vertex, replacing it with a face.
    For an icosahedron, this creates a truncated icosahedron (soccer ball).

    Args:
        mesh: Input polyhedron mesh
        truncation_factor: How much to truncate (0-1, typically 0.33)

    Returns:
        Truncated polyhedron mesh
    """
    # Build vertex-to-faces and vertex-to-neighbors mappings
    vertex_faces = [[] for _ in range(len(mesh.vertices))]
    vertex_neighbors = [set() for _ in range(len(mesh.vertices))]

    for face_idx, face in enumerate(mesh.faces):
        for i, v in enumerate(face):
            vertex_faces[v].append(face_idx)
            # Add neighbors (previous and next vertex in face)
            prev_v = face[(i - 1) % len(face)]
            next_v = face[(i + 1) % len(face)]
            vertex_neighbors[v].add(prev_v)
            vertex_neighbors[v].add(next_v)

    # Create new vertices along each edge
    new_vertices = []
    vertex_map = {}  # Maps (v0, v1) -> new_vertex_idx

    edges_seen = set()
    for v0 in range(len(mesh.vertices)):
        for v1 in vertex_neighbors[v0]:
            if (v0, v1) not in edges_seen and (v1, v0) not in edges_seen:
                # Create two new vertices on this edge
                p0 = mesh.vertices[v0]
                p1 = mesh.vertices[v1]

                new_v0 = p0 + truncation_factor * (p1 - p0)
                new_v1 = p1 + truncation_factor * (p0 - p1)

                idx0 = len(new_vertices)
                new_vertices.append(new_v0)
                vertex_map[(v0, v1)] = idx0

                idx1 = len(new_vertices)
                new_vertices.append(new_v1)
                vertex_map[(v1, v0)] = idx1

                edges_seen.add((v0, v1))

    new_vertices = np.array(new_vertices)

    # Create faces - these will be polygons (not triangles)
    polygon_faces = []

    # 1. Pentagon/hexagon faces from truncating vertices
    for v_idx in range(len(mesh.vertices)):
        neighbors = list(vertex_neighbors[v_idx])

        # Order neighbors by rotation around the vertex
        # Use the faces to get proper ordering
        ordered_neighbors = []
        for face_idx in vertex_faces[v_idx]:
            face = mesh.faces[face_idx]
            v_pos = list(face).index(v_idx)
            prev_v = face[(v_pos - 1) % len(face)]
            next_v = face[(v_pos + 1) % len(face)]

            if prev_v not in ordered_neighbors:
                ordered_neighbors.append(prev_v)
            if next_v not in ordered_neighbors:
                ordered_neighbors.append(next_v)

        # Create face from the new vertices on edges to these neighbors
        face_verts = [vertex_map[(v_idx, n)] for n in ordered_neighbors if (v_idx, n) in vertex_map]
        if len(face_verts) >= 3:
            polygon_faces.append(face_verts)

    # 2. Hexagon faces from truncating original triangle faces
    for face in mesh.faces:
        face_verts = []
        for i in range(len(face)):
            v0 = face[i]
            v1 = face[(i + 1) % len(face)]
            if (v0, v1) in vertex_map:
                face_verts.append(vertex_map[(v0, v1)])
        if len(face_verts) >= 3:
            polygon_faces.append(face_verts)

    # Return PolyhedronMesh which handles both polygon and triangulated representation
    return PolyhedronMesh(new_vertices, polygon_faces)


def project_to_sphere(mesh, radius: float = 1.0):
    """
    Project a polyhedron onto a sphere.

    Each vertex is projected to the surface of a sphere with the given radius.
    The topology (faces) remains the same.

    Args:
        mesh: Input polyhedron mesh (PolyhedronMesh or trimesh.Trimesh)
        radius: Radius of the sphere

    Returns:
        New mesh with vertices projected onto sphere (same type as input)
    """
    # Normalize each vertex to lie on unit sphere, then scale by radius
    vertices = mesh.vertices
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    spherical_vertices = (vertices / norms) * radius

    # Return same type as input
    if isinstance(mesh, PolyhedronMesh):
        return PolyhedronMesh(spherical_vertices, mesh.faces)
    else:
        return trimesh.Trimesh(vertices=spherical_vertices, faces=mesh.faces)


def get_face_groups(mesh, tolerance: float = 1e-6) -> dict:
    """
    Group faces by shape (based on edge lengths and angles).

    This identifies unique panel types in the spherical polyhedron.

    Args:
        mesh: The mesh (PolyhedronMesh or trimesh.Trimesh, should be projected to sphere)
        tolerance: Tolerance for comparing face shapes

    Returns:
        Dictionary mapping group_id -> list of face indices
    """
    # Get faces list (works for both PolyhedronMesh and trimesh.Trimesh)
    if isinstance(mesh, PolyhedronMesh):
        faces = mesh.faces
    else:
        faces = mesh.faces.tolist()

    face_signatures = []

    for face_idx, face in enumerate(faces):
        # Get vertices of this face
        verts = mesh.vertices[face]

        # Calculate edge lengths
        n_verts = len(verts)
        edge_lengths = []
        for i in range(n_verts):
            v0 = verts[i]
            v1 = verts[(i + 1) % n_verts]
            length = np.linalg.norm(v1 - v0)
            edge_lengths.append(length)

        # Sort edge lengths to get a canonical representation
        edge_lengths = sorted(edge_lengths)

        # Calculate face area (simple polygon area in 3D)
        area = _calculate_polygon_area_3d(verts)

        # Create signature: (n_edges, edge_lengths..., area)
        signature = (n_verts, *edge_lengths, area)
        face_signatures.append((face_idx, signature))

    # Group faces by similar signatures
    groups = {}
    group_id = 0

    for face_idx, sig in face_signatures:
        # Find matching group
        matched = False
        for gid, (group_sig, _) in groups.items():
            # Check if signatures match within tolerance
            if len(sig) == len(group_sig):
                if all(abs(a - b) < tolerance for a, b in zip(sig, group_sig)):
                    groups[gid][1].append(face_idx)
                    matched = True
                    break

        if not matched:
            groups[group_id] = (sig, [face_idx])
            group_id += 1

    # Return simplified format: group_id -> face_indices
    return {gid: faces for gid, (_, faces) in groups.items()}


def _calculate_polygon_area_3d(vertices: np.ndarray) -> float:
    """Calculate area of a 3D polygon using cross product method."""
    if len(vertices) < 3:
        return 0.0

    # Sum of cross products
    total = np.zeros(3)
    for i in range(len(vertices)):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % len(vertices)]
        total += np.cross(v0, v1)

    # Area is half the magnitude
    return 0.5 * np.linalg.norm(total)


def list_available_polyhedra() -> list[str]:
    """List all available polyhedra."""
    return [
        'tetrahedron',
        'cube',
        'octahedron',
        'dodecahedron',
        'icosahedron',
        'truncated_icosahedron',
    ]
