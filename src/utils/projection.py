"""3D to 2D projection utilities for plotter output."""
import numpy as np
from typing import Tuple


def project_3d_to_2d(data: np.ndarray, elevation: float = 30.0, azimuth: float = -60.0) -> np.ndarray:
    """Project 3D data to 2D using rotation matrix + drop depth.

    Args:
        data: (N, 3) array
        elevation: degrees, angle above XY plane
        azimuth: degrees, rotation around Z axis
    Returns:
        (N, 2) array
    """
    # Convert to radians
    el = np.radians(elevation)
    az = np.radians(azimuth)

    # Rotation matrices
    cos_el, sin_el = np.cos(el), np.sin(el)
    cos_az, sin_az = np.cos(az), np.sin(az)

    # Combined rotation: first azimuth around Z, then elevation around X
    Rz = np.array([[cos_az, -sin_az, 0], [sin_az, cos_az, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cos_el, -sin_el], [0, sin_el, cos_el]])
    R = Rx @ Rz

    rotated = data @ R.T
    return rotated[:, :2]  # drop depth


def find_optimal_projection(data: np.ndarray, n_trials: int = 100) -> Tuple[float, float, np.ndarray]:
    """Find projection angles that maximize 2D convex hull area.

    Returns (best_elevation, best_azimuth, projected_data).
    """
    from scipy.spatial import ConvexHull

    best_area = 0
    best_el, best_az = 30.0, -60.0
    best_proj = None

    for _ in range(n_trials):
        el = np.random.uniform(-90, 90)
        az = np.random.uniform(-180, 180)
        proj = project_3d_to_2d(data, el, az)
        try:
            hull = ConvexHull(proj)
            if hull.volume > best_area:  # volume = area in 2D
                best_area = hull.volume
                best_el, best_az = el, az
                best_proj = proj
        except Exception:
            continue

    return best_el, best_az, best_proj if best_proj is not None else project_3d_to_2d(data)
