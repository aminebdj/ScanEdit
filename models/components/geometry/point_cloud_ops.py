"""
Point Cloud Operations

Functions for projecting, analyzing, and processing 3D point clouds.
Includes region analysis for orientation detection and plane projection.
"""

import numpy as np
import open3d as o3d


def project_point_cloud_to_plane(point_cloud, plane_normal, visualize=False):
    """
    Project 3D points onto plane defined by normal vector through origin.
    
    Removes the component of each point in the direction of the plane normal,
    effectively "flattening" the point cloud onto the plane.
    
    Math:
        For point p and plane normal n:
        projection = p - (p · n)n
        
    Used for:
        - Computing 2D footprints of objects
        - Analyzing symmetry by projecting onto dominant face
        - Finding object outlines
    
    Args:
        point_cloud (np.ndarray): Points to project, shape (N, 3)
        plane_normal (np.ndarray): Plane normal vector, shape (3,)
        visualize (bool): If True, show original and projected points
    
    Returns:
        np.ndarray: Projected points, shape (N, 3)
        
    Example:
        >>> points = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normal = np.array([0, 0, 1])  # XY plane
        >>> projected = project_point_cloud_to_plane(points, normal)
        >>> # Z coordinates become 0
    """
    # Normalize plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Compute dot product of each point with normal
    dot_products = np.dot(point_cloud, plane_normal)

    # Remove normal component: p_proj = p - (p·n)n
    projected_points = point_cloud - np.outer(dot_products, plane_normal)
    
    if visualize:
        visualize_point_clouds(point_cloud, projected_points)

    return projected_points


def find_max_adjacent_regions(points, visualize=False):
    """
    Find the pair of adjacent regions with maximum combined point density.
    
    Divides the point cloud's bounding box into 8 regions (like octants) and
    finds which two adjacent regions have the most points combined. Used to
    determine object orientation - the densest adjacent regions typically
    indicate the "front" or main functional side of an object.
    
    Algorithm:
        1. Compute bounding box and divide into 8 regions
        2. Count points in each region
        3. Check predefined adjacency pairs
        4. Return centroid of the pair with most points
    
    Args:
        points (np.ndarray): Point cloud, shape (N, 3)
        visualize (bool): If True, show regions with best pair highlighted
    
    Returns:
        np.ndarray: Centroid of best region pair, shape (3,)
        
    Used by:
        - reset_dominant_normal() to determine object front direction
        - set_object_scene_related_infos() for orientation
    """
    # Compute bounding box
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_dims = bbox_max - bbox_min
    bbox_center = (bbox_max + bbox_min) / 2
    
    # Adjust center to upper 80% (focus on top of object)
    bbox_center[-1] = bbox_min[-1] + 0.8 * bbox_dims[-1]

    # Define 8 regions (only using top 4 for furniture)
    x_mid, y_mid, z_mid = bbox_center
    regions = {
        'top-left-front': (
            np.array([bbox_min[0], bbox_min[1], z_mid]),
            np.array([x_mid, y_mid, bbox_max[2]])
        ),
        'top-right-front': (
            np.array([x_mid, bbox_min[1], z_mid]),
            np.array([bbox_max[0], y_mid, bbox_max[2]])
        ),
        'top-left-back': (
            np.array([bbox_min[0], y_mid, z_mid]),
            np.array([x_mid, bbox_max[1], bbox_max[2]])
        ),
        'top-right-back': (
            np.array([x_mid, y_mid, z_mid]),
            np.array([bbox_max[0], bbox_max[1], bbox_max[2]])
        ),
    }

    # Count points in each region
    region_counts = {}
    region_points = {}
    for region, bounds in regions.items():
        min_bound, max_bound = bounds
        mask = np.all(points > min_bound, axis=-1) & np.all(points < max_bound, axis=-1)
        region_counts[region] = mask.sum()
        region_points[region] = points[mask]

    # Define which regions are adjacent
    adjacency = [
        ('top-left-front', 'top-right-front'),
        ('top-left-front', 'top-left-back'),
        ('top-right-front', 'top-right-back'),
        ('top-left-back', 'top-right-back')
    ]

    # Find adjacent pair with most points
    max_points = 0
    best_pair = None
    for region1, region2 in adjacency:
        combined_count = region_counts[region1] + region_counts[region2]
        if combined_count > max_points:
            max_points = combined_count
            best_pair = (region1, region2)

    # Optional visualization
    if visualize:
        region_colors = {r: [1, 0, 0] for r in regions.keys()}
        region_colors[best_pair[0]] = [0, 1, 0]
        region_colors[best_pair[1]] = [0, 1, 0]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        geometries = [point_cloud]

        for region, (min_corner, max_corner) in regions.items():
            box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                o3d.geometry.AxisAlignedBoundingBox(min_corner, max_corner)
            )
            box.paint_uniform_color(region_colors[region])
            geometries.append(box)

        o3d.visualization.draw_geometries(geometries)

    # Return centroid of best pair
    try:
        best_pair_points = np.concatenate([
            region_points[best_pair[0]],
            region_points[best_pair[1]]
        ])
        return best_pair_points.mean(axis=0)
    except:
        # Fallback if regions are empty
        return points.mean(axis=0)


def visualize_point_clouds(original_points, projected_points):
    """
    Visualize original and projected point clouds side by side.
    
    Debug utility for checking projection results.
    
    Args:
        original_points (np.ndarray): Original points, shape (N, 3)
        projected_points (np.ndarray): Projected points, shape (N, 3)
    """
    # Create Open3D point clouds
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.paint_uniform_color([1, 0, 0])  # Red

    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
    projected_pcd.paint_uniform_color([0, 1, 0])  # Green

    o3d.visualization.draw_geometries([original_pcd, projected_pcd])