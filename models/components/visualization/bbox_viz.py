"""
Bounding Box Visualization

Functions for visualizing 3D bounding boxes, arrows, and geometric primitives
in Open3D. Used for debugging and scene understanding visualization.
"""

import numpy as np
import open3d as o3d


def create_sphere(radius=0.05, color=[1, 0, 0]):
    """
    Create colored sphere mesh with normals.
    
    Args:
        radius (float): Sphere radius in meters. Default: 0.05
        color (list): RGB color [0-1]. Default: red
    
    Returns:
        o3d.geometry.TriangleMesh: Sphere mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere


def create_arrow(start, end, color=[0, 0, 1]):
    """
    Create arrow from start to end point.
    
    Useful for visualizing normals, directions, and orientations.
    
    Args:
        start (np.ndarray): Arrow start position, shape (3,)
        end (np.ndarray): Arrow end position, shape (3,)
        color (list): RGB color [0-1]. Default: blue
    
    Returns:
        o3d.geometry.TriangleMesh: Arrow mesh
        
    Example:
        >>> arrow = create_arrow([0, 0, 0], [1, 0, 0], color=[1, 0, 0])
        >>> # Red arrow pointing in +X direction
    """
    length = np.linalg.norm(np.array(end) - np.array(start))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.03, 
        cone_radius=0.06,
        cylinder_height=0.5,
        cone_height=0.1
    )
    
    # Compute rotation to align with desired direction
    direction = (np.array(end) - np.array(start)) / length
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    angle = np.arccos(np.dot(z_axis, direction))
    
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        arrow.rotate(R, center=(0, 0, 0))
    
    arrow.translate(start)
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()
    return arrow


def create_lines(corners, color=[0, 1, 0]):
    """
    Create wireframe box from 8 corner points.
    
    Args:
        corners (np.ndarray): 8 corners of box, shape (8, 3)
        color (list): RGB line color [0-1]. Default: green
    
    Returns:
        o3d.geometry.LineSet: Wireframe box
        
    Note:
        Corner order matters - should follow standard bounding box convention:
        Bottom face: 0-3, Top face: 4-7
    """
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def create_cylinder(start_point, end_point, color, radius=0.005, resolution=20):
    """
    Create cylinder connecting two points.
    
    Useful for visualizing connections, paths, or trajectories.
    
    Args:
        start_point (np.ndarray): Start position, shape (3,)
        end_point (np.ndarray): End position, shape (3,)
        color (list): RGB color [0-1]
        radius (float): Cylinder radius. Default: 0.005
        resolution (int): Mesh resolution. Default: 20
    
    Returns:
        o3d.geometry.TriangleMesh: Cylinder mesh
    """
    height = np.linalg.norm(end_point - start_point)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=resolution
    )
    
    # Move to start point
    cylinder.translate(start_point)
    
    # Rotate to align with end point
    direction = (end_point - start_point) / height
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([
        np.arccos(direction[2]), 
        0, 
        np.arctan2(direction[1], direction[0])
    ])
    cylinder.rotate(rotation_matrix, center=start_point)
    cylinder.paint_uniform_color(color)
    
    return cylinder


def visualize_bboxes(objects, sphere_colors, arrow_colors, get_geometrie=False):
    """
    Visualize objects as oriented bounding boxes with orientation arrows.
    
    Each object is shown as:
    - Wireframe box (8 corners connected)
    - Corner spheres
    - Orientation arrow showing front direction
    
    Args:
        objects (list of dict): Object dicts with keys:
                               'id', 'name', 'dimensions', 'base', 'orientation'
        sphere_colors (list): RGB color for spheres/wireframe
        arrow_colors (list): RGB color for orientation arrow
        get_geometrie (bool): If True, return geometries instead of displaying
    
    Returns:
        list or None: Geometries if get_geometrie=True, else displays and returns None
        
    Example:
        >>> objects = [{'dimensions': [1, 0.5, 0.8], 'base': [0, 0, 0], 'orientation': 45, ...}]
        >>> visualize_bboxes(objects, [1,0,0], [0,1,0])
    """
    vis_objects = []
    
    for obj in objects:
        obj_id, name, dimensions, base, orientation = obj.values()
        
        w, h, d = dimensions
        cx, cy, min_z = base
        orientation_rad = np.radians(orientation)
        
        # Define corner offsets (unrotated)
        offsets = np.array([
            [-w/2, -h/2, 0], [w/2, -h/2, 0],
            [w/2, h/2, 0], [-w/2, h/2, 0],
            [-w/2, -h/2, d], [w/2, -h/2, d],
            [w/2, h/2, d], [-w/2, h/2, d]
        ])
        
        # Rotation matrix (Z-axis rotation)
        R = np.array([
            [np.cos(orientation_rad), -np.sin(orientation_rad), 0],
            [np.sin(orientation_rad), np.cos(orientation_rad), 0],
            [0, 0, 1]
        ])
        
        # Compute rotated corners
        corners = [R @ offset + np.array([cx, cy, min_z]) for offset in offsets]
        
        # Add spheres at corners
        for corner in corners:
            sphere = create_sphere(color=sphere_colors)
            sphere.translate(corner)
            vis_objects.append(sphere)
        
        # Add wireframe
        vis_objects.append(create_lines(corners, color=sphere_colors))
        
        # Add orientation arrow
        arrow_start = np.array([cx, cy, min_z + d/2])
        arrow_end = arrow_start + np.array([
            np.cos(orientation_rad), 
            np.sin(orientation_rad), 
            0
        ]) * (w * 2)
        vis_objects.append(create_arrow(arrow_start, arrow_end, color=arrow_colors))

    if get_geometrie:
        return vis_objects
    o3d.visualization.draw_geometries(vis_objects)


def visualize_bboxes_from_point_cloud(objects, sphere_colors, arrow_colors, 
                                      get_geometrie=False):
    """
    Visualize point clouds as axis-aligned bounding boxes.
    
    Similar to visualize_bboxes but computes bbox from point cloud
    rather than using predefined dimensions. No orientation arrows.
    
    Args:
        objects (list of np.ndarray): Point clouds, each (N, 3)
        sphere_colors (list): RGB color for visualization
        arrow_colors (list): Unused (kept for API compatibility)
        get_geometrie (bool): If True, return geometries
    
    Returns:
        list or None: Geometries if requested, else displays
    """
    vis_objects = []
    
    for obj in objects:
        # Compute bbox from point cloud
        dimensions = obj.max(axis=0) - obj.min(axis=0)
        base = (obj.max(axis=0) + obj.min(axis=0)) / 2
        base[-1] = obj[:, -1].min()  # Base at minimum Z
        
        w, h, d = dimensions
        cx, cy, min_z = base
        
        # Corner offsets
        offsets = np.array([
            [-w/2, -h/2, 0], [w/2, -h/2, 0],
            [w/2, h/2, 0], [-w/2, h/2, 0],
            [-w/2, -h/2, d], [w/2, -h/2, d],
            [w/2, h/2, d], [-w/2, h/2, d]
        ])
        
        # Compute corners (no rotation)
        corners = [offset + np.array([cx, cy, min_z]) for offset in offsets]
        
        # Add spheres
        for corner in corners:
            sphere = create_sphere(color=sphere_colors)
            sphere.translate(corner)
            vis_objects.append(sphere)
        
        # Add wireframe
        vis_objects.append(create_lines(corners, color=sphere_colors))

    if get_geometrie:
        return vis_objects
    o3d.visualization.draw_geometries(vis_objects)


def visualize_bboxes_from_bbox_min_max(bboxes, sphere_colors, arrow_colors, 
                                       get_geometrie=False):
    """
    Visualize bounding boxes specified as (min_corner, max_corner) pairs.
    
    Args:
        bboxes (list of tuple): Each tuple is (min_corner, max_corner)
                               where each corner is shape (3,)
        sphere_colors (list): RGB color
        arrow_colors (list): Unused
        get_geometrie (bool): If True, return geometries
    
    Returns:
        list or None: Geometries if requested
    """
    vis_objects = []
    
    for min_corner, max_corner in bboxes:
        # Compute dimensions and base
        w, h, d = max_corner - min_corner
        cx, cy, min_z = (max_corner + min_corner) / 2

        # Corner offsets
        offsets = np.array([
            [-w/2, -h/2, 0], [w/2, -h/2, 0],
            [w/2, h/2, 0], [-w/2, h/2, 0],
            [-w/2, -h/2, d], [w/2, -h/2, d],
            [w/2, h/2, d], [-w/2, h/2, d]
        ])

        corners = [offset + np.array([cx, cy, min_z]) for offset in offsets]

        # Add spheres
        for corner in corners:
            sphere = create_sphere(radius=0.02, color=sphere_colors)
            sphere.translate(corner)
            vis_objects.append(sphere)
        
        # Add wireframe
        vis_objects.append(create_lines(corners, color=sphere_colors))

    if get_geometrie:
        return vis_objects
    o3d.visualization.draw_geometries(vis_objects)