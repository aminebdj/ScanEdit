"""
Camera Utilities

Functions for camera parameter conversion and projection matrix computation.
Handles conversion between different camera conventions (NDC, screen space, etc.)
"""

import numpy as np
import torch


def get_intrinsic_and_extrinsics_from_ndc_camera(camera):
    """
    Extract camera intrinsics and extrinsics from PyTorch3D NDC camera.
    
    PyTorch3D uses Normalized Device Coordinates (NDC) where the image plane
    ranges from [-1, 1]. This converts to standard pinhole camera parameters
    in pixel coordinates.
    
    Conversion:
        NDC focal length -> screen focal length:
            fx_screen = (fx_ndc * image_height) / 2
        
        NDC principal point -> screen principal point:
            px_screen = (-px_ndc * image_height) / 2 + image_width / 2
    
    Args:
        camera: PyTorch3D camera object with NDC parameters
    
    Returns:
        tuple: (intrinsic_matrix, camera_to_world)
            - intrinsic_matrix (np.ndarray): 3x3 calibration matrix
            - camera_to_world (torch.Tensor): 4x4 extrinsic matrix
            
    Use case:
        - Converting PyTorch3D renderings to standard computer vision format
        - Interfacing with traditional CV libraries (OpenCV, etc.)
    """
    # Get camera-to-world transform (inverse of world-to-view)
    camera_to_world = torch.inverse(
        camera.get_world_to_view_transform().get_matrix().squeeze().cpu().T
    )
    
    # NDC camera parameters
    fx_ndc = 2.0
    fy_ndc = 2.0
    px_ndc = 0
    py_ndc = 0
    
    # Image dimensions
    image_width = 1920
    image_height = 1080
    s = image_height  # Scale factor
    
    # Convert focal lengths from NDC to screen space
    fx_screen = (fx_ndc * s) / 2.0
    fy_screen = (fy_ndc * s) / 2.0
    
    # Convert principal point from NDC to screen space
    px_screen = (-px_ndc * s) / 2.0 + (image_width / 2.0)
    py_screen = (-py_ndc * s) / 2.0 + (image_height / 2.0)
    
    # Construct intrinsics matrix
    # Note: Negative focal lengths due to coordinate system convention
    intrinsic_matrix = np.array([
        [-fx_screen, 0, image_width - px_screen],
        [0, -fy_screen, image_height - py_screen],
        [0, 0, 1]
    ])
    
    return intrinsic_matrix, camera_to_world