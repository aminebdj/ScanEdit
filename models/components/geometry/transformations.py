"""
Geometric Transformations

Functions for computing rotation matrices, group transformations, and
coordinate system conversions. Core math for object placement and orientation.
"""

import numpy as np


def get_individual_transformations(objects, x_group, y_group, theta_group):
    """
    Decompose group transformation into individual object transformations.
    
    When multiple objects are transformed as a group (e.g., moving a dining set),
    this computes the resulting position and orientation of each object.
    
    Math:
        1. Compute group center: centroid of all objects
        2. For each object:
            - Find position relative to group center
            - Rotate relative position by group rotation
            - Add group translation
            - Result is new object position
    
    Args:
        objects (list of np.ndarray): List of point clouds, each (N, 3)
        x_group (float): Group translation in X
        y_group (float): Group translation in Y
        theta_group (float): Group rotation around Z axis (radians)
    
    Returns:
        list of dict: [{'x': x_i, 'y': y_i, 'theta': theta_i}] for each object
        
    Example:
        >>> obj1 = np.array([[0, 0, 0], [1, 0, 0]])  # Chair
        >>> obj2 = np.array([[2, 0, 0], [3, 0, 0]])  # Table
        >>> # Move group by (5, 5) and rotate 90 degrees
        >>> transforms = get_individual_transformations(
        ...     [obj1, obj2], 5, 5, np.pi/2
        ... )
    """
    # Compute group center (centroid of all objects combined)
    all_points = np.vstack(objects)
    group_center = np.mean(all_points, axis=0)[:2]  # XY only

    results = []
    for obj in objects:
        # Object center before transformation
        obj_center = np.mean(obj, axis=0)[:2]

        # Position relative to group center
        rel_x, rel_y = obj_center - group_center

        # Rotate relative position by group rotation
        cos_t, sin_t = np.cos(theta_group), np.sin(theta_group)
        rel_x_new = cos_t * rel_x - sin_t * rel_y
        rel_y_new = sin_t * rel_x + cos_t * rel_y

        # New object center = rotated_relative_pos + group_translation + group_center - obj_center
        new_obj_center = (
            np.array([rel_x_new, rel_y_new]) + 
            np.array([x_group, y_group]) + 
            group_center - obj_center
        )

        # Object inherits group rotation
        theta_i = theta_group

        results.append({
            'x': new_obj_center[0], 
            'y': new_obj_center[1], 
            'theta': theta_i
        })

    return results


def rotation_matrix(axis, theta):
    """
    Create 3D rotation matrix using Rodrigues' rotation formula.
    
    Rotates points around an arbitrary axis by angle theta.
    
    Math (Rodrigues' formula):
        R = I + sin(θ)K + (1-cos(θ))K²
        where K is the skew-symmetric matrix of the axis vector
    
    Args:
        axis (np.ndarray): Rotation axis, 3D vector (will be normalized)
        theta (float): Rotation angle in radians
    
    Returns:
        np.ndarray: 3x3 rotation matrix
        
    Example:
        >>> # Rotate 90 degrees around Z axis
        >>> R = rotation_matrix(np.array([0, 0, 1]), np.pi/2)
        >>> point = np.array([1, 0, 0])
        >>> rotated = R @ point  # [0, 1, 0]
    """
    # Normalize axis vector
    axis = axis / np.linalg.norm(axis)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta

    # Rodrigues' rotation formula
    R = np.array([
        [
            cos_theta + axis[0]**2 * one_minus_cos,
            axis[0] * axis[1] * one_minus_cos - axis[2] * sin_theta,
            axis[0] * axis[2] * one_minus_cos + axis[1] * sin_theta
        ],
        [
            axis[1] * axis[0] * one_minus_cos + axis[2] * sin_theta,
            cos_theta + axis[1]**2 * one_minus_cos,
            axis[1] * axis[2] * one_minus_cos - axis[0] * sin_theta
        ],
        [
            axis[2] * axis[0] * one_minus_cos - axis[1] * sin_theta,
            axis[2] * axis[1] * one_minus_cos + axis[0] * sin_theta,
            cos_theta + axis[2]**2 * one_minus_cos
        ]
    ])
    
    return R


def get_rotation_matrix(face_normal, target_axis=np.array([0, 1, 0])):
    """
    Compute rotation matrix to align face_normal with target_axis.
    
    Finds the rotation that transforms face_normal to point in the same
    direction as target_axis. Used for reorienting objects.
    
    Math:
        1. Rotation axis: cross(face_normal, target_axis)
        2. Rotation angle: arccos(dot(face_normal, target_axis))
        3. Apply Rodrigues' formula
    
    Args:
        face_normal (np.ndarray): Normal vector to align, shape (3,)
        target_axis (np.ndarray): Target direction, shape (3,). Default: [0,1,0]
    
    Returns:
        np.ndarray: 3x3 rotation matrix
        
    Example:
        >>> # Align object's front (facing +X) to face +Y
        >>> normal = np.array([1, 0, 0])
        >>> R = get_rotation_matrix(normal, np.array([0, 1, 0]))
    """
    # Normalize both vectors
    face_normal = face_normal / np.linalg.norm(face_normal)
    target_axis = target_axis / np.linalg.norm(target_axis)

    # Compute rotation axis (perpendicular to both vectors)
    rotation_axis = np.cross(face_normal, target_axis)
    
    # Check if vectors are already aligned
    if np.linalg.norm(rotation_axis) < 1e-6:
        return np.eye(3)  # No rotation needed

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Compute rotation angle
    cos_theta = np.dot(face_normal, target_axis)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Rodrigues' rotation formula via skew-symmetric matrix
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    I = np.eye(3)
    rotation_matrix = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return rotation_matrix


def perpendicular_axis(v):
    """
    Compute perpendicular vector in XY plane.
    
    Returns a vector perpendicular to v, constrained to the XY plane.
    Used for finding rotation axes that keep objects upright.
    
    Math:
        For vector [x, y, z], perpendicular in XY is [-y, x, 0]
    
    Args:
        v (np.ndarray): Input 3D vector
    
    Returns:
        np.ndarray: Perpendicular vector in XY plane
        
    Example:
        >>> v = np.array([1, 0, 5])
        >>> perp = perpendicular_axis(v)  # [0, 1, 0]
    """
    return np.array([-v[1], v[0], 0])