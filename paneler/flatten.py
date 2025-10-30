"""Panel flattening for spherical polygons."""

import numpy as np
import trimesh
from scipy.spatial.distance import pdist, squareform


def flatten_spherical_face(vertices: np.ndarray, method: str = 'stereographic') -> np.ndarray:
    """
    Flatten a spherical polygon to 2D.

    Args:
        vertices: Array of 3D points on sphere surface (N x 3)
        method: Flattening method ('stereographic' or 'azimuthal')

    Returns:
        Array of 2D points (N x 2)
    """
    if method == 'stereographic':
        return _stereographic_projection(vertices)
    elif method == 'azimuthal':
        return _azimuthal_equidistant(vertices)
    else:
        raise ValueError(f"Unknown flattening method: {method}")


def _stereographic_projection(vertices: np.ndarray) -> np.ndarray:
    """
    Stereographic projection from sphere to plane.

    Projects from south pole onto plane at equator.
    This preserves angles but not distances.

    Args:
        vertices: 3D points on sphere (N x 3)

    Returns:
        2D points (N x 2)
    """
    # Rotate so face center is at north pole
    center = np.mean(vertices, axis=0)
    center = center / np.linalg.norm(center)

    # Create rotation matrix to align center with z-axis
    z_axis = np.array([0, 0, 1])
    rotation = _rotation_matrix_from_vectors(center, z_axis)

    # Rotate vertices
    rotated = vertices @ rotation.T

    # Apply stereographic projection
    # Project from south pole (0, 0, -1) onto plane z=0
    # Formula: (x, y) = (X/(1-Z), Y/(1-Z))
    x = rotated[:, 0] / (1 - rotated[:, 2])
    y = rotated[:, 1] / (1 - rotated[:, 2])

    return np.column_stack([x, y])


def _azimuthal_equidistant(vertices: np.ndarray) -> np.ndarray:
    """
    Azimuthal equidistant projection.

    Preserves distances from center point.
    Good for small patches on sphere.

    Args:
        vertices: 3D points on sphere (N x 3)

    Returns:
        2D points (N x 2)
    """
    # Get center of face
    center = np.mean(vertices, axis=0)
    center = center / np.linalg.norm(center)

    # Rotate so center is at north pole
    z_axis = np.array([0, 0, 1])
    rotation = _rotation_matrix_from_vectors(center, z_axis)
    rotated = vertices @ rotation.T

    # Convert to spherical coordinates
    # For points near north pole, use azimuthal equidistant
    x = rotated[:, 0]
    y = rotated[:, 1]
    z = rotated[:, 2]

    # Distance from north pole (angular distance)
    c = np.arccos(np.clip(z, -1, 1))

    # Azimuthal angle
    azimuth = np.arctan2(y, x)

    # Azimuthal equidistant: radius is proportional to angular distance
    r = c

    # Convert to Cartesian
    x_2d = r * np.cos(azimuth)
    y_2d = r * np.sin(azimuth)

    return np.column_stack([x_2d, y_2d])


def _rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Find rotation matrix that aligns vec1 to vec2.

    Args:
        vec1: Source vector (3,)
        vec2: Target vector (3,)

    Returns:
        Rotation matrix (3, 3)
    """
    # Normalize vectors
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    # Calculate rotation axis and angle
    v = np.cross(a, b)
    c = np.dot(a, b)

    # Handle case where vectors are parallel
    if np.allclose(v, 0):
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation - find perpendicular axis
            if abs(a[0]) < abs(a[1]):
                perp = np.array([1, 0, 0])
            else:
                perp = np.array([0, 1, 0])
            v = np.cross(a, perp)
            v = v / np.linalg.norm(v)
            return _rotation_matrix_from_axis_angle(v, np.pi)

    s = np.linalg.norm(v)

    # Rodrigues' rotation formula
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create rotation matrix from axis and angle.

    Args:
        axis: Rotation axis (3,), should be normalized
        angle: Rotation angle in radians

    Returns:
        Rotation matrix (3, 3)
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])


def add_seam_allowance(points_2d: np.ndarray, allowance: float) -> np.ndarray:
    """
    Add seam allowance around a 2D polygon.

    Offsets the polygon boundary outward by the allowance distance.

    Args:
        points_2d: Polygon vertices in 2D (N x 2)
        allowance: Distance to offset outward

    Returns:
        New polygon with seam allowance (N x 2)
    """
    if allowance <= 0:
        return points_2d

    n = len(points_2d)
    offset_points = []

    for i in range(n):
        # Get three consecutive points
        p_prev = points_2d[(i - 1) % n]
        p_curr = points_2d[i]
        p_next = points_2d[(i + 1) % n]

        # Edge vectors
        edge_prev = p_curr - p_prev
        edge_next = p_next - p_curr

        # Normalize
        edge_prev = edge_prev / np.linalg.norm(edge_prev)
        edge_next = edge_next / np.linalg.norm(edge_next)

        # Perpendicular vectors (pointing outward)
        perp_prev = np.array([-edge_prev[1], edge_prev[0]])
        perp_next = np.array([-edge_next[1], edge_next[0]])

        # Average perpendicular direction
        perp_avg = (perp_prev + perp_next) / 2
        perp_avg = perp_avg / np.linalg.norm(perp_avg)

        # Offset point
        offset = p_curr + perp_avg * allowance
        offset_points.append(offset)

    return np.array(offset_points)


def get_panel_info(vertices_3d: np.ndarray, vertices_2d: np.ndarray) -> dict:
    """
    Calculate panel information and measurements.

    Args:
        vertices_3d: Original 3D vertices on sphere
        vertices_2d: Flattened 2D vertices

    Returns:
        Dictionary with panel measurements
    """
    n = len(vertices_2d)

    # Edge lengths (in 2D flattened space)
    edge_lengths = []
    for i in range(n):
        p0 = vertices_2d[i]
        p1 = vertices_2d[(i + 1) % n]
        length = np.linalg.norm(p1 - p0)
        edge_lengths.append(length)

    # Area (2D)
    area_2d = _polygon_area(vertices_2d)

    # Geodesic edge lengths (on sphere)
    geodesic_lengths = []
    for i in range(n):
        p0 = vertices_3d[i]
        p1 = vertices_3d[(i + 1) % n]
        # Angular distance on unit sphere
        cos_angle = np.dot(p0, p1) / (np.linalg.norm(p0) * np.linalg.norm(p1))
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        geodesic_lengths.append(angle)

    return {
        'n_sides': n,
        'edge_lengths_2d': edge_lengths,
        'geodesic_lengths': geodesic_lengths,
        'area_2d': area_2d,
        'perimeter_2d': sum(edge_lengths),
    }


def _polygon_area(points: np.ndarray) -> float:
    """
    Calculate area of 2D polygon using shoelace formula.

    Args:
        points: Polygon vertices (N x 2)

    Returns:
        Area
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
