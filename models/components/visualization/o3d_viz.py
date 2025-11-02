"""
Open3D Visualization Utilities

High-level visualization functions combining meshes, point clouds, and annotations.
"""

import numpy as np
import open3d as o3d
from visualization.bbox_viz import create_sphere


def visualize_data_with_target(target_point, target_normal, data, mask):
    """
    Visualize target point, normal, and masked mesh region.
    
    Shows:
    - Red sphere at target point
    - Green arrow for target normal
    - Mesh with only masked region visible
    
    Used for debugging surface alignment and placement targets.
    
    Args:
        target_point (np.ndarray): 3D target coordinates, shape (3,)
        target_normal (np.ndarray): Target normal vector, shape (3,)
        data (o3d.geometry.TriangleMesh): Mesh to visualize
        mask (np.ndarray): Boolean vertex mask, shape (N,)
    """
    # Create target point sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.paint_uniform_color([1, 0, 0])  # Red
    sphere.translate(target_point)

    # Create normal arrow
    line_points = [target_point, target_point + target_normal * 0.2]
    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
    
    # Create masked mesh
    data_new = o3d.geometry.TriangleMesh()
    vertices = np.asarray(data.vertices)
    triangles = np.asarray(data.triangles)
    vertex_colors = np.asarray(data.vertex_colors)
    vertex_normals = np.asarray(data.vertex_normals)

    # Zero out masked vertices
    vertices[~mask] = 0
    vertex_colors[~mask] = 0
    vertex_normals[~mask] = 0

    # Keep only triangles with all vertices in mask
    mask_triangles = np.all(mask[triangles], axis=1)
    triangles_filtered = triangles[mask_triangles]

    data_new.vertices = o3d.utility.Vector3dVector(vertices)
    data_new.triangles = o3d.utility.Vector3iVector(triangles_filtered)
    data_new.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    data_new.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    
    # Visualize
    o3d.visualization.draw_geometries(
        [sphere, line_set, data_new],
        window_name="Target Point, Normal, and Vertices",
        width=800,
        height=600,
        point_show_normal=False
    )