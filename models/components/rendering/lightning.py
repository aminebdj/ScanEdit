"""
Lighting and Shading Utilities

Functions for computing lighting effects in rendering.
"""

import torch


def specular(points, normals, direction, color, camera_position, shininess):
    """
    Compute specular lighting using Phong reflection model.
    
    Specular highlights simulate shiny surfaces by reflecting light
    toward the camera based on surface normal.
    
    Math (Phong Model):
        reflect_dir = 2(n · l)n - l
        specular = (reflect_dir · view_dir)^shininess
        
        where:
        - n: surface normal
        - l: light direction
        - view_dir: direction to camera
        - shininess: specular exponent (higher = sharper highlight)
    
    Args:
        points (torch.Tensor): Surface points, shape (N, 3)
        normals (torch.Tensor): Surface normals, shape (N, 3)
        direction (torch.Tensor): Light direction, shape (3,) or (N, 3)
        color (torch.Tensor): Light color, shape (3,) or (N, 3)
        camera_position (torch.Tensor): Camera position, shape (3,)
        shininess (float): Specular exponent (typical range: 10-200)
    
    Returns:
        torch.Tensor: Specular color contribution, shape (N, 3)
        
    Example:
        >>> points = torch.rand(1000, 3)
        >>> normals = torch.rand(1000, 3)
        >>> light_dir = torch.tensor([0, 0, -1])
        >>> spec = specular(points, normals, light_dir, 
        ...                 torch.tensor([1, 1, 1]), 
        ...                 torch.tensor([0, 0, 5]), 
        ...                 shininess=32)
    """
    # Compute reflection direction
    dot_nl = (normals * direction).sum(dim=-1, keepdim=True)
    reflect_direction = 2 * dot_nl * normals - direction
    
    # Compute view direction
    view_direction = torch.nn.functional.normalize(
        camera_position - points, dim=-1
    )
    
    # Compute specular intensity
    specular_intensity = torch.clamp(
        (reflect_direction * view_direction).sum(dim=-1, keepdim=True),
        min=0
    ) ** shininess
    
    return color * specular_intensity