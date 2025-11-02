"""
Transformation Utilities

Functions for computing transformations between object poses and
converting between different pose representations.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import ast


def get_transformation(prev_pose, new_pose):
    """
    Compute 4x4 transformation matrix between two poses.
    
    Given previous and new poses (center + orientation), computes the
    transformation that maps from prev_pose to new_pose coordinate frame.
    
    Math:
        T = T_trans @ T_prev_trans @ T_rot @ T_prev_trans^-1
        where:
        - T_rot: Rotation from prev to new orientation
        - T_trans: Translation from prev to new center
        - T_prev_trans: Move to prev center (for rotation about center)
    
    Args:
        prev_pose (tuple): (center, orientation) where:
                          - center: [x, y, z]
                          - orientation: [rx, ry, rz] Euler angles
        new_pose (tuple): (center, orientation) in same format
    
    Returns:
        np.ndarray: 4x4 transformation matrix
        
    Example:
        >>> prev = ([0, 0, 0], [0, 0, 0])
        >>> new = ([1, 0, 0], [0, 0, np.pi/2])  # Move 1m in X, rotate 90° around Z
        >>> T = get_transformation(prev, new)
    """
    prev_center, prev_orientation = prev_pose
    new_center, new_orientation = new_pose
    
    # Create rotation matrices
    prev_rotation_matrix = R.from_euler('xyz', prev_orientation).as_matrix()
    new_rotation_matrix = R.from_euler('xyz', new_orientation).as_matrix()
    
    # Compute relative rotation
    rotation_matrix = new_rotation_matrix @ np.linalg.inv(prev_rotation_matrix)
    
    # Compute translation
    translation_vector = np.array(new_center) - np.array(prev_center)
    
    # Construct transformation matrices
    transformation_matrix_trans = np.eye(4)
    transformation_matrix_rotation = np.eye(4)
    transformation_matrix_prev_trans = np.eye(4)
    
    transformation_matrix_rotation[:3, :3] = rotation_matrix
    transformation_matrix_trans[:3, 3] = translation_vector
    transformation_matrix_prev_trans[:3, 3] = np.array(prev_center)
    
    # Combine: translate to origin, rotate, translate back, then translate to new position
    transformation_matrix = (
        transformation_matrix_trans @ 
        transformation_matrix_prev_trans @ 
        transformation_matrix_rotation @ 
        np.linalg.inv(transformation_matrix_prev_trans)
    )
    
    return transformation_matrix


def compute_prev_next_transformation(class_agnostic_preds, scene_description, groq):
    """
    Compute object transformations from string-encoded scene descriptions.
    
    Parses pipe-separated scene descriptions and computes transformation
    for each object from previous to next state.
    
    Args:
        class_agnostic_preds: Unused (kept for API compatibility)
        scene_description (str): Format "obj1;center;orientation|obj2;..."
        groq (str): Target state in same format
    
    Returns:
        dict: {object_name: transformation_matrix}
        
    Example:
        >>> scene = "chair;[0,0,0];[0,0,0]|table;[1,0,0];[0,0,0]"
        >>> target = "chair;[1,0,0];[0,0,1.57]|table;[1,0,0];[0,0,0]"
        >>> transforms = compute_prev_next_transformation(None, scene, target)
    """
    # Parse scene descriptions
    scene_description = {
        i.split(';')[0]: i.split(';')[1:] 
        for i in scene_description.split('|')
    }
    groq = {
        i.split(';')[0]: i.split(';')[1:] 
        for i in groq.split('|')
    }
    
    object_transformations = {}
    valid_keys = list(scene_description.keys())
    if '' in valid_keys:
        valid_keys.remove('')
    
    for k in groq.keys():
        if k not in valid_keys:
            continue
        
        # Parse center and orientation
        prev_pose = (
            ast.literal_eval(scene_description[k][0]),
            ast.literal_eval(scene_description[k][2])
        )
        new_pose = (
            ast.literal_eval(groq[k][0]),
            ast.literal_eval(groq[k][2])
        )
        
        object_transformations[k] = get_transformation(prev_pose, new_pose)
    
    return object_transformations


def compute_prev_next_transformation_dict(previous, next):
    """
    Compute transformations between scene graph dictionaries.
    
    Args:
        previous (dict): Previous scene state {id: {'3d_center': {x, y, z}, ...}}
        next (dict): Next scene state in same format
    
    Returns:
        tuple: (transformations_dict, changed_ids)
            - transformations_dict: {id: 4x4 matrix}
            - changed_ids: List of object IDs that moved
            
    Example:
        >>> prev = {0: {'3d_center': {'x': 0, 'y': 0, 'z': 0}}}
        >>> next = {0: {'3d_center': {'x': 1, 'y': 0, 'z': 0}}}
        >>> transforms, ids = compute_prev_next_transformation_dict(prev, next)
    """
    object_transformations = {}
    ids = []
    
    for k in next.keys():
        # Get centers (no rotation)
        prev_pose = (list(previous[k]['3d_center'].values()), [0, 0, 0])
        new_pose = (list(next[k]['3d_center'].values()), [0, 0, 0])
        
        object_transformations[k] = get_transformation(prev_pose, new_pose)
        
        # Track which objects actually moved
        if (object_transformations[k] != np.eye(4)).any():
            ids.append(int(k))
    
    return object_transformations, ids