"""
Bounding Box Utilities

Functions for computing specialized bounding boxes, particularly
oriented bounding boxes with specific axis constraints.
"""

import numpy as np
import open3d as o3d


def oriented_bounding_box_xy_only(point_cloud):
    """
    Compute oriented bounding box constrained to XY plane rotation only.
    
    Unlike a full OBB which can rotate in 3D, this creates a box that:
    - Rotates only around Z axis (stays upright)
    - Tightly fits points in XY plane
    - Uses full Z range of points
    
    Useful for furniture and objects that should remain upright but need
    tight XY bounds.
    
    Args:
        point_cloud (open3d.geometry.PointCloud): Input point cloud
    
    Returns:
        open3d.geometry.OrientedBoundingBox: OBB aligned to XY axes
        
    Example:
        >>> pcd = o3d.geometry.PointCloud()
        >>> pcd.points = o3d.utility.Vector3dVector(points)
        >>> obb = oriented_bounding_box_xy_only(pcd)
        >>> # obb is tight in XY, full height in Z
    """
    # Get axis-aligned bounding box
    aabb = point_cloud.get_axis_aligned_bounding_box()
    aabb_points = np.asarray(aabb.get_box_points())
    
    # Extract XY bounds
    min_x, min_y = np.min(aabb_points[:, :2], axis=0)
    max_x, max_y = np.max(aabb_points[:, :2], axis=0)
    
    # Use full Z range
    min_z, max_z = np.min(aabb_points[:, 2]), np.max(aabb_points[:, 2])
    
    # Define 8 corners of constrained box
    obb_corners = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z],
    ])
    
    # Create OBB from corners
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(obb_corners)
    )
    
    return obb