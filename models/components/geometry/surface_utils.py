import numpy as np
import torch
import open3d as o3d
import cv2
import copy
from scipy.spatial import Delaunay
from models.components.utils.geometry_utils import rotation_matrix_from_vector_to_z


class SurfaceUtils:
    """Utilities for surface detection and processing"""
    
    @staticmethod
    def get_top_support_surfaces_as_pointcloud(obj_points, bin_size=0.1, top_n=7):
        """
        Identify top N support surfaces based on z-density
        
        Args:
            obj_points: N x 3 array of points
            bin_size: Size of bins for z-distribution
            top_n: Number of top surfaces to return
            
        Returns:
            Concatenated point cloud of top support surfaces
        """
        z_vals = obj_points[:, 2]
        hist, bin_edges = np.histogram(
            z_vals,
            bins=np.arange(z_vals.min(), z_vals.max() + bin_size, bin_size)
        )
        
        top_bin_indices = np.argsort(hist)[-top_n:][::-1]
        
        support_surfaces = []
        for idx in top_bin_indices:
            z_min, z_max = bin_edges[idx], bin_edges[idx + 1]
            support_points = obj_points[(z_vals >= z_min) & (z_vals < z_max)]
            support_surfaces.append(support_points)
            
        if support_surfaces:
            return np.vstack(support_surfaces)
        else:
            return np.array([[np.inf, np.inf, np.inf]])
            
    @staticmethod
    def segment_largest_plane_and_align(surface_points, distance_threshold=0.01,
                                       ransac_n=3, num_iterations=1000, scene_center=None):
        """
        Segment largest planar component using RANSAC
        
        Args:
            surface_points: Input points
            distance_threshold: RANSAC distance threshold
            ransac_n: Number of points for RANSAC
            num_iterations: RANSAC iterations
            scene_center: Center of scene
            
        Returns:
            Tuple of (inlier_mask, rotation_matrix, surface_normal)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        [a, b, c, d] = plane_model
        surface_normal = np.array([a, b, c])
        
        if scene_center is not None:
            surface_normal = surface_normal * np.sign(
                (-surface_points.mean(0) + scene_center).T @ surface_normal
            )
        surface_normal /= np.linalg.norm(surface_normal)
        
        R = rotation_matrix_from_vector_to_z(surface_normal)
        
        inlier_mask = np.zeros(len(surface_points), dtype=bool)
        inlier_mask[inliers] = True
        
        return inlier_mask, R, surface_normal
        
    @staticmethod
    def create_surface_mesh_from_contour(contour_points, contour_normals,
                                         floor_height, R, n_points=50000):
        """
        Create surface mesh from contour points
        
        Args:
            contour_points: Contour boundary points
            contour_normals: Normals at contour points
            floor_height: Height of surface
            R: Rotation matrix
            n_points: Number of points to sample
            
        Returns:
            O3D triangle mesh
        """
        # Create initial triangulation
        faces = []
        num_points = len(contour_points)
        for i in range(num_points - 2):
            p1 = 0
            p2 = i + 1
            p3 = i + 2
            faces.append([p1, p2, p3])
            
        faces = np.array(faces)
        
        # Create mesh and sample points
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(contour_points)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        
        point_cloud = mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(point_cloud.points)
        
        # 2D Delaunay triangulation
        tri = Delaunay(points[:, :2])
        
        # Create final mesh
        mesh = o3d.geometry.TriangleMesh()
        points = np.concatenate([
            points[:, :2],
            np.ones((len(points), 1)) * floor_height
        ], axis=-1)
        faces = tri.simplices
        
        # Filter faces inside contour
        edge_centers = (points[faces[:, 0]] + points[faces[:, 1]] + points[faces[:, 2]]) / 3
        dir_vecs = edge_centers[:, None, :2] - contour_points[None, :, :2]
        in_contour = (dir_vecs * contour_normals[None, :, :2]).sum(axis=-1)
        min_dist = np.argmin(np.linalg.norm(dir_vecs, axis=-1), axis=-1)
        in_contour_mask = (np.take_along_axis(in_contour, min_dist[:, None], axis=1) > 0)
        faces = faces[~(in_contour_mask.squeeze())]
        
        mesh.vertices = o3d.utility.Vector3dVector(points @ R)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        
        return mesh
        
    @staticmethod
    def extract_wall_contour_from_grid(vertices, mask, grid_resolution=0.01):
        """
        Extract wall contour using grid-based approach
        
        Args:
            vertices: Vertex positions (rotated to align with axes)
            mask: Mask for wall vertices
            grid_resolution: Grid cell size
            
        Returns:
            Tuple of (contour_points, contour_normals)
        """
        # Create grid
        min_xy = vertices.min(axis=0)[:2]
        grid_xy = np.unique(((vertices[:, :2] - min_xy) // grid_resolution).astype(int), axis=0)
        max_xy = grid_xy.max(axis=0)
        
        grid = np.zeros((max_xy[0] + 1, max_xy[1] + 1), dtype=bool)
        grid[grid_xy[:, 0], grid_xy[:, 1]] = True
        
        # Smooth and threshold
        grid = cv2.convertScaleAbs(grid * 255)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        grid_smoothed = cv2.dilate(grid, kernel)
        thresh = cv2.threshold(grid_smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        point_count_per_contour = [len(cnt) for cnt in contours]
        max_points_ind = point_count_per_contour.index(max(point_count_per_contour))
        contour_cells = contours[max_points_ind].squeeze().tolist()
        
        # Convert to world coordinates
        converted_points = []
        for y, x in contour_cells:
            cell_center_x = x * grid_resolution + min_xy[0]
            cell_center_y = y * grid_resolution + min_xy[1]
            converted_points.append([cell_center_x, cell_center_y])
            
        contour_points = np.array(converted_points)
        
        # Add interpolation points
        num_points = 4
        contour_points = np.concatenate([
            contour_points,
            np.stack([
                contour_points[-1] + ((i + 1) / (num_points + 1)) * (contour_points[0] - contour_points[-1])
                for i in range(num_points)
            ])
        ])
        
        # Calculate normals
        tangents = np.gradient(contour_points, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        
        normals = np.empty_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]
        
        return contour_points, normals