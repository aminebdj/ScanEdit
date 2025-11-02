"""
Debug Plotting Utilities

Matplotlib and Open3D-based plotting for debugging geometric computations.
Not used in production - only for development and troubleshooting.
"""

import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import open3d as o3d


def plot_normals(train_normal, target_normal, name='alignment'):
    """
    Plot two 2D vectors to visualize alignment.
    
    Used for debugging normal alignment during optimization.
    Shows learned normal vs target normal as arrows from origin.
    
    Args:
        train_normal (torch.Tensor): Learned normal vector, shape (2,)
        target_normal (np.ndarray): Target normal vector, shape (2,)
        name (str): Plot title. Default: 'alignment'
        
    Example:
        >>> learned = torch.tensor([1.0, 0.5])
        >>> target = np.array([1.0, 0.0])
        >>> plot_normals(learned, target, name='Chair Front Normal')
    """
    origin = np.array([0, 0])
    
    # Convert to numpy
    vector1 = train_normal.cpu().detach().numpy()
    vector2 = target_normal.numpy()

    # Create plot
    plt.figure()
    plt.quiver(*origin, *vector1, color='r', scale=1, scale_units='xy', 
              angles='xy', label='trained normal')
    plt.quiver(*origin, *vector2, color='b', scale=1, scale_units='xy', 
              angles='xy', label='target normal')

    # Configure plot
    plt.xlim(-1, 5)
    plt.ylim(-1, 5)
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.grid()
    plt.title(f'{name}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_polygon_and_points(polygon_points, points):
    """
    Plot 2D polygon vertices and reference points.
    
    Visualizes polygon boundary (filled) and test points to debug
    inside/outside calculations and signed distance computations.
    
    Args:
        polygon_points (torch.Tensor): Polygon vertices, shape (N, 2)
        points (torch.Tensor): Reference points to test, shape (M, 2)
        
    Example:
        >>> polygon = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        >>> test_pts = torch.tensor([[0.5, 0.5], [2, 2]])
        >>> plot_polygon_and_points(polygon, test_pts)
        >>> # Saves to ./debug.png
    """
    # Convert to numpy
    polygon_points_np = polygon_points.numpy()
    points_np = points.numpy()

    # Plot polygon with fill
    plt.plot(*zip(*polygon_points_np), marker='o', color='blue', 
            ls='-', label='Polygon Vertices')
    plt.fill(*zip(*polygon_points_np), color='blue', alpha=0.1)

    # Plot reference points
    plt.scatter(points_np[:, 0], points_np[:, 1], color='red', 
               marker='x', label='Reference Points')
    
    # Configure plot
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.title('Polygon and Reference Points in 2D')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('./debug.png')
    plt.close()  # Close to avoid memory issues


def plot_contour_with_normals(contour_points, contour_normals):
    """
    Plot polygon contour with normal vectors as arrows.
    
    Visualizes boundary with inward-pointing normals. Useful for debugging
    signed distance calculations and surface orientation.
    
    Args:
        contour_points (torch.Tensor): Contour vertices, shape (N, 2)
        contour_normals (torch.Tensor): Normals at each vertex, shape (N, 2)
        
    Example:
        >>> points = torch.tensor([[0,0], [1,0], [1,1], [0,1]])
        >>> normals = torch.tensor([[0,1], [1,0], [0,-1], [-1,0]])
        >>> plot_contour_with_normals(points, normals)
    """
    plt.figure(figsize=(8, 8))
    plt.plot(contour_points[:, 0], contour_points[:, 1], 'o-', label='Polygon')

    # Draw normal arrows at each vertex
    for i, point in enumerate(contour_points[:-1]):
        plt.arrow(
            point[0], point[1], 
            contour_normals[i][0] * 0.1, 
            contour_normals[i][1] * 0.1,
            head_width=0.02, head_length=0.02, fc='r', ec='r'
        )

    # Highlight vertices
    plt.scatter(contour_points[:, 0], contour_points[:, 1], 
               c='blue', label='Contour Points')
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plot_polygon_with_point(polygon, points):
    """
    Plot polygon with points, sorting vertices by polar angle.
    
    Ensures polygon is drawn correctly by sorting vertices around centroid.
    Useful for visualizing convex hulls and extracted boundaries.
    
    Args:
        polygon (torch.Tensor): Polygon vertices, shape (N, 2 or 3)
        points (torch.Tensor): Points to overlay, shape (M, 2 or 3)
        
    Example:
        >>> polygon = torch.tensor([[1,0], [0,1], [1,1], [0,0]])  # Unsorted
        >>> points = torch.tensor([[0.5, 0.5]])
        >>> plot_polygon_with_point(polygon, points)
        >>> # Polygon drawn with vertices sorted by angle
    """
    # Use only XY coordinates
    if polygon.shape[-1] == 3:
        polygon = polygon[:, :2]
    
    # Compute centroid
    cent = polygon.mean(dim=0).tolist()
    polygon = polygon.tolist()
    
    # Sort by polar angle around centroid
    polygon.sort(key=lambda p: math.atan2(p[1] - cent[1], p[0] - cent[0]))
    polygon = torch.tensor(polygon, dtype=points.dtype, device=points.device)
    
    # Plot polygon and points
    pylab.scatter([p[0] for p in polygon.cpu().tolist()],
                  [p[1] for p in polygon.cpu().tolist()])
    pylab.scatter([p[0] for p in points.cpu().tolist()],
                  [p[1] for p in points.cpu().tolist()])
    
    # Draw polygon outline
    pylab.gca().add_patch(
        patches.Polygon(polygon.cpu().tolist(), closed=False, fill=False)
    )
    pylab.grid()
    pylab.show()


def visualize_mask(depth_image, mask, point_2d=None):
    """
    Visualize depth image with mask overlay side-by-side.
    
    Shows original depth image and masked version for debugging
    segmentation, visibility, or surface detection.
    
    Args:
        depth_image (np.ndarray): Depth image to visualize
        mask (np.ndarray): Binary or weighted mask, same shape as depth_image
        point_2d (tuple, optional): (x, y) coordinates to highlight with red dot
        
    Example:
        >>> depth = np.random.rand(480, 640)
        >>> mask = depth > 0.5
        >>> visualize_mask(depth, mask, point_2d=(100, 200))
    """
    plt.figure(figsize=(12, 6))
    
    # Display original depth image
    plt.subplot(1, 2, 1)
    plt.imshow(depth_image, cmap='gray')
    plt.title("Depth Image")
    plt.axis("off")
    
    # Display with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)  # Transparent overlay
    
    # Optional point highlight
    if point_2d:
        plt.scatter(*point_2d, c='red', s=100, label='Point 2D')
        plt.legend()
    
    plt.title("Mask Overlay")
    plt.axis("off")
    
    plt.show()


def show_3d_points_with_mesh(mesh, out, skip_k=5):
    """
    Visualize visible surface points overlaid on mesh.
    
    Used for debugging visibility calculations and surface sampling.
    Shows which points are deemed visible from a particular viewpoint
    as red spheres on the mesh.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Base mesh to display
        out (dict): Output from visibility computation with keys:
                   - 'surface': All surface points, shape (N, 3)
                   - 'mask_in_view_visible': Boolean mask, shape (N,)
        skip_k (int): Visualize every k-th point for performance. Default: 5
    
    Example:
        >>> mesh = o3d.io.read_triangle_mesh("chair.obj")
        >>> visibility_result = compute_visibility(mesh, camera)
        >>> show_3d_points_with_mesh(mesh, visibility_result, skip_k=10)
        >>> # Opens 3D viewer with mesh and visible points as spheres
    
    Note:
        Creates sphere mesh for each point - can be slow with many points.
        Increase skip_k to improve performance.
    """
    from geometry.mesh_operations import create_spheres_with_normals
    
    # Extract visible points
    visible_points = out['surface'][out['mask_in_view_visible']]
    
    # Create sphere meshes (subsample with skip_k for performance)
    points_mesh = create_spheres_with_normals(
        visible_points[::skip_k],
        radius=0.02
    )
    
    # Display mesh + visible points together
    o3d.visualization.draw_geometries(points_mesh + [mesh])