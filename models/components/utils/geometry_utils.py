"""
Geometric Utility Functions

(Existing docstring and imports...)
"""

import torch
import numpy as np
from scipy.spatial import ConvexHull


def max_point_resolution(point_cloud_1, point_cloud_2):
    """
    Compute maximum point spacing (resolution) between two point clouds.
    
    Finds the maximum of the minimum nearest-neighbor distances in both clouds.
    Used as a reference scale for collision detection thresholds.
    
    Math:
        For each point, find distance to its nearest neighbor
        resolution = max(max(min_dists_1), max(min_dists_2))
    
    Args:
        point_cloud_1 (torch.Tensor): First point cloud, shape (N, 3)
        point_cloud_2 (torch.Tensor): Second point cloud, shape (M, 3)
    
    Returns:
        float: Maximum resolution (largest nearest-neighbor distance)
        
    Example:
        >>> pc1 = torch.rand(100, 3)
        >>> pc2 = torch.rand(150, 3)
        >>> res = max_point_resolution(pc1, pc2)
        >>> # res indicates typical spacing between points
    """
    # Compute pairwise distances within each cloud
    dists_1 = torch.cdist(point_cloud_1, point_cloud_1)
    dists_2 = torch.cdist(point_cloud_2, point_cloud_2)
    
    # Ignore self-distances
    dists_1.fill_diagonal_(float('inf'))
    dists_2.fill_diagonal_(float('inf'))
    
    # Find minimum distance to nearest neighbor for each point
    min_dists_1, _ = dists_1.min(dim=1)
    min_dists_2, _ = dists_2.min(dim=1)
    
    # Return maximum resolution
    max_resolution = max(min_dists_1.max().item(), min_dists_2.max().item())
    
    return max_resolution


def batched_norm(vector_diff, batch_size):
    """
    Compute vector norms in batches for memory efficiency.
    
    When computing norms of many vectors, processing in batches
    prevents out-of-memory errors.
    
    Args:
        vector_diff (torch.Tensor): Vectors to compute norms, shape (..., N, D)
        batch_size (int): Number of vectors to process per batch
    
    Returns:
        torch.Tensor: Norms with shape (..., N)
        
    Example:
        >>> vectors = torch.rand(100000, 3)
        >>> norms = batched_norm(vectors, batch_size=10000)
    """
    num_batches = (vector_diff.shape[0] + batch_size - 1) // batch_size
    results = []

    for i in range(num_batches):
        batch = vector_diff[i * batch_size : (i + 1) * batch_size]
        batch_norm = torch.norm(batch, dim=-1)
        results.append(batch_norm)

    return torch.cat(results, dim=0)


def grid_based_density_cluster(points, grid_size=0.1):
    """
    Find highest-density cluster using grid-based approach.
    
    Divides space into uniform grid cells and returns points from
    the cell with most points. Useful for finding collision regions.
    
    Algorithm:
        1. Quantize points to grid cells
        2. Count points per cell
        3. Return points in densest cell
    
    Args:
        points (torch.Tensor): Point cloud, shape (N, D)
        grid_size (float): Size of grid cells in meters
    
    Returns:
        torch.Tensor: Points in highest-density cluster
        
    Example:
        >>> points = torch.rand(1000, 3)
        >>> cluster = grid_based_density_cluster(points, grid_size=0.5)
        >>> # cluster contains points from densest 0.5m x 0.5m x 0.5m cell
    """
    # Remove duplicate points
    points = torch.unique(points, dim=0)
    
    # Quantize to grid indices
    grid_indices = (points / grid_size).floor().long()  # (N, D)

    # Count points per grid cell
    grid_keys, counts = torch.unique(grid_indices, dim=0, return_counts=True)

    # Find densest cell
    max_density_idx = torch.argmax(counts)
    max_density_key = grid_keys[max_density_idx]

    # Extract points in densest cell
    cluster_mask = torch.all(grid_indices == max_density_key, dim=1)
    cluster_points = points[cluster_mask]

    return cluster_points


def extract_polygon(point_cloud, n_furthest_points):
    """
    Extract polygon boundary from point cloud using convex hull.
    
    Finds convex hull of points, then selects n points on the hull
    that are furthest from the centroid. Creates a simplified polygon
    boundary representation.
    
    Algorithm:
        1. Project to 2D (XY plane)
        2. Compute convex hull
        3. Find centroid of hull points
        4. Select n furthest points from centroid
    
    Args:
        point_cloud (torch.Tensor): 3D points, shape (N, 3)
        n_furthest_points (int): Number of polygon vertices
    
    Returns:
        torch.Tensor: Polygon vertices, shape (n_furthest_points, 3)
                     Returns None if point cloud is degenerate
        
    Example:
        >>> points = torch.rand(1000, 3)
        >>> polygon = extract_polygon(points, n_furthest_points=8)
        >>> # polygon has 8 vertices approximating the boundary
    
    Use case:
        Creating simplified boundaries for surface detection
    """
    # Check for degenerate case (all points identical)
    if len(point_cloud.unique(dim=0)) == 1:
        return None
    
    # Project to 2D (XY plane)
    points_2d = point_cloud[:, :2].cpu().numpy()

    # Compute convex hull
    hull = ConvexHull(points_2d)
    hull_points = point_cloud[hull.vertices]

    # Compute centroid of hull
    centroid = torch.mean(hull_points, dim=0)

    # Find n furthest points from centroid
    distances = torch.norm(hull_points - centroid, dim=1)
    _, indices = torch.topk(distances, k=n_furthest_points)

    # Extract polygon vertices
    polygon_points = hull_points[indices]

    return polygon_points


def rotation_matrix_from_vector_to_z(normal, target=np.array([0, 0, 1])):
    """
    Compute rotation matrix aligning normal with target vector.
    
    Uses Rodrigues' rotation formula to find rotation that maps
    normal to target direction. Default target is +Z axis.
    
    Math:
        1. Rotation axis: cross(normal, target)
        2. Rotation angle: arccos(dot(normal, target))
        3. Rodrigues: R = I + sin(θ)K + (1-cos(θ))K²
        
    Args:
        normal (np.ndarray): Vector to align, shape (3,)
        target (np.ndarray): Target direction, shape (3,). Default: [0,0,1]
    
    Returns:
        np.ndarray: 3x3 rotation matrix
        
    Example:
        >>> normal = np.array([1, 0, 0])
        >>> R = rotation_matrix_from_vector_to_z(normal)
        >>> rotated = R @ normal  # Now points toward +Z
    """
    # Normalize input
    normal = normal / np.linalg.norm(normal)

    # Compute rotation axis
    axis = np.cross(normal, target)
    
    # Check if already aligned
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Compute angle
    angle = np.arccos(np.dot(normal, target))
    
    # Skew-symmetric matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Rodrigues' formula
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R