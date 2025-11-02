"""
PCA and Statistical Analysis Utilities

Functions for Principal Component Analysis and statistical computations
on point clouds. Used for finding dominant directions and analyzing distributions.
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_pca_for_face(face_points):
    """
    Compute principal component (dominant direction) for a set of points.
    
    Uses PCA to find the direction of maximum variance, which typically
    corresponds to the longest dimension of a face or surface patch.
    
    Math:
        PCA finds eigenvectors of covariance matrix.
        First component = direction of maximum variance
    
    Args:
        face_points (np.ndarray): Points on a face/surface, shape (N, 3)
    
    Returns:
        np.ndarray: Dominant direction vector, shape (3,)
        
    Example:
        >>> face_points = np.array([[0,0,0], [1,0,0], [2,0,0]])
        >>> direction = compute_pca_for_face(face_points)
        >>> # Returns approximately [1, 0, 0] (X-axis dominant)
        
    Use case:
        Finding orientation of table surfaces, wall segments, etc.
    """
    pca = PCA(n_components=1)  # Only need dominant direction
    pca.fit(face_points)
    return pca.components_[0]


def compute_symmetry_score(projected_points):
    """
    Compute symmetry score based on point distribution around center.
    
    Measures how evenly points are distributed around their mean/median.
    Lower scores indicate more symmetric distributions.
    
    Algorithm:
        1. Compute mean and median of points
        2. Calculate average distance to mean
        3. Calculate average distance to median
        4. Score = (mean_diff + median_diff) / 2
    
    Args:
        projected_points (np.ndarray): Points to analyze, shape (N, D)
    
    Returns:
        float: Symmetry score (lower = more symmetric)
        
    Example:
        >>> # Symmetric distribution
        >>> points = np.array([[-1,0], [1,0], [0,-1], [0,1]])
        >>> score = compute_symmetry_score(points)
        >>> # Low score indicates symmetry
        
    Use case:
        Evaluating how symmetric an object is after projection
    """
    # Calculate center using mean and median
    mean = np.mean(projected_points, axis=0)
    median = np.median(projected_points, axis=0)
    
    # Average distance to mean
    mean_diff = np.linalg.norm(projected_points - mean, axis=1).mean()
    
    # Average distance to median
    median_diff = np.linalg.norm(projected_points - median, axis=1).mean()
    
    # Combined symmetry score (lower is more symmetrical)
    return (mean_diff + median_diff) / 2