"""
Object Point Cloud Module

This module contains the OBJ_POINT_CLOUD class which represents individual objects
in the 3D scene. Each object stores point cloud data, surfaces, normals, and
various geometric properties needed for scene understanding and manipulation.

Key Features:
- Point cloud representation with normals
- Oriented bounding box calculation
- Surface detection and storage
- Dominant normal computation for object orientation
- Spatial relationship tracking
"""

import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from PIL import Image
import random
import copy
import torch.nn as nn
from pytorch3d.ops import box3d_overlap
import cv2
from scipy.interpolate import splprep, splev
import fpsample
# Optimization
from models.components.optimization.losses import inside_polygon_loss

# External libraries (no change)
from scipy.spatial import ConvexHull, Delaunay

# Geometry utilities
from models.components.utils.geometry_utils import rotation_matrix_from_vector_to_z
from models.components.geometry.bounding_box import oriented_bounding_box_xy_only
from models.components.geometry.point_cloud_ops import (
    project_point_cloud_to_plane,
    find_max_adjacent_regions
)
from models.components.geometry.transformations import (
    get_rotation_matrix,
    rotation_matrix,
    perpendicular_axis
)

# Symmetry analysis
from models.components.utils.pca_utils import compute_symmetry_score

# Visualization
from models.components.visualization.bbox_viz import create_sphere, create_lines
from models.components.utils.color_utils import generate_vibrant_cool_colors

class TransformationNetwork(nn.Module):
    """
    Neural network module for learning object transformations.
    
    This network learns translation (x, y, z) and rotation (theta) parameters
    for placing objects in the scene. Used during optimization to find valid
    object placements that satisfy spatial constraints.
    
    Attributes:
        x, y, z: Translation parameters
        theta: Rotation angle around z-axis
    """
    
    def __init__(self, x_init=0.0, y_init=0.0, z_init=0.0, theta_init=0.0, device='cuda'):
        super(TransformationNetwork, self).__init__()
        self.device = device
        self.x = nn.Parameter(torch.tensor(x_init, device=self.device), requires_grad=True)
        self.y = nn.Parameter(torch.tensor(y_init, device=self.device), requires_grad=True)
        self.z = nn.Parameter(torch.tensor(z_init, device=self.device), requires_grad=True)
        self.theta = nn.Parameter(torch.tensor(theta_init, device=self.device), requires_grad=True)

    def print_params(self):
        """Print current rotation angle in degrees"""
        print(f"Theta is {(self.theta / np.pi) * 180} degrees")
        
    def get_transformation(self):
        """
        Get 4x4 transformation matrix from learned parameters
        
        Returns:
            numpy.ndarray: 4x4 homogeneous transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = np.array([
            [np.cos(self.theta.detach().cpu().item()), -np.sin(self.theta.detach().cpu().item()), 0],
            [np.sin(self.theta.detach().cpu().item()), np.cos(self.theta.detach().cpu().item()), 0],
            [0, 0, 1]
        ])
        T[0, -1] = self.x.detach().cpu().item()
        T[1, -1] = self.y.detach().cpu().item()
        T[2, -1] = self.z.detach().cpu().item()
        return T
    
    def forward(self, x, center=None, detach_theta=False):
        """
        Apply transformation to points
        
        Args:
            x: Input points (N x 3)
            center: Optional center for rotation
            detach_theta: Whether to detach theta gradient
            
        Returns:
            Transformed points
        """
        x = self.forward_xy(x, center, detach_theta)
        return x.squeeze()
    
    def forward_xy(self, x, center=None, detach_theta=False):
        """Apply XY translation and rotation"""
        if x.device != self.device:
            x = x.to(self.device)
        translation = (self.x * torch.tensor([1, 0, 0], device=self.device)[None, :] +
                      self.y * torch.tensor([0, 1, 0], device=self.device)[None, :] +
                      self.z.detach() * torch.tensor([0, 0, 1], device=self.device)[None, :])
        return (self.forward_xy_rotate_around_center(x + translation.detach(), center=center, 
                                                     detach_theta=detach_theta) - 
                translation.detach() + translation)
    
    def forward_xy_rotate_around_center(self, x, center=None, detach_theta=False):
        """Rotate points around a center"""
        if x.device != self.device:
            x = x.to(self.device)
        if len(x.shape) == 1:
            x = x[None, :]
        if torch.norm(x) == 0:
            return x
        
        obj_center = x.mean(dim=0)[None, :] if center is None else center[None, :]
        x = self.forward_xy_rotate(x - obj_center, detach_theta) + obj_center
        return x
        
    def forward_xy_rotate(self, x, detach_theta=False):
        """Apply rotation in XY plane"""
        if x.device != self.device:
            x = x.to(self.device)
        if len(x.shape) == 1:
            x = x[None, :]
        theta = self.theta.detach() if detach_theta else self.theta
        x = ((x[:, 0][:, None] * torch.cos(theta) - x[:, 1][:, None] * torch.sin(theta)) * 
             torch.tensor([1, 0, 0], device=self.device)[None, :] +
             (x[:, 0][:, None] * torch.sin(theta) + x[:, 1][:, None] * torch.cos(self.theta)) * 
             torch.tensor([0, 1, 0], device=self.device)[None, :] +
             x[:, 2][:, None] * torch.tensor([0, 0, 1], device=self.device)[None, :])
        return x.squeeze()

    def forward_xy_translate(self, x):
        """Apply XY translation only"""
        if x.device != self.device:
            x = x.to(self.device)
        if len(x.shape) == 1:
            x = x[None, :]
        x = ((x[:, 0][:, None] + self.x) * torch.tensor([1, 0, 0], device=self.device)[None, :] +
             (x[:, 1][:, None] + self.y) * torch.tensor([0, 1, 0], device=self.device)[None, :] +
             (x[:, 2][:, None] + self.z.detach()) * torch.tensor([0, 0, 1], device=self.device)[None, :])
        return x.squeeze()
    
    def forward_z(self, x):
        """Apply Z translation only"""
        if x.device != self.device:
            x = x.to(self.device)
        if len(x.shape) == 1:
            x = x[None, :]
        x = (x[:, 0][:, None] * torch.tensor([1, 0, 0], device=self.device)[None, :] +
             x[:, 1][:, None] * torch.tensor([0, 1, 0], device=self.device)[None, :] +
             (self.z + x[:, 2][:, None]) * torch.tensor([0, 0, 1], device=self.device)[None, :])
        return x.squeeze()


class OBJ_POINT_CLOUD:
    """
    Represents a 3D object as a point cloud with additional geometric properties.
    
    This class handles all object-level data including:
    - Point cloud representation
    - Oriented bounding box
    - Surface detection and storage
    - Normal computation
    - Spatial relationships with other objects
    - Rendering metadata (projections, regions, etc.)
    
    The object can be in two states:
    1. Regular object with full geometry
    2. Corrupt object (invalid/too small) - only stores ID and name
    """
    
    def __init__(self, np_array=None, normals=None, obj_name=None, scene_center=None,
                 cameras=None, scene_name='unknown', scene_points=None, R=None,
                 obj_id=None, empty=False, wall_objects=None, corrupt=False, annos=None):
        """
        Initialize object point cloud.
        
        Args:
            np_array: Point cloud as numpy array (N x 3)
            normals: Normal vectors for each point (N x 3)
            obj_name: Semantic name of object
            scene_center: Center of the scene for orientation computation
            cameras: Camera objects for rendering/projection
            scene_name: Name of the scene
            scene_points: Full scene point cloud (for context)
            R: Optional rotation matrix
            obj_id: Unique identifier for object
            empty: If True, create empty object for later loading
            wall_objects: List of wall objects (for orientation computation)
            corrupt: If True, mark as invalid object
            annos: Optional annotations
        """
        # Handle corrupt/invalid objects
        if obj_name == 'invalid' or corrupt:
            self.obj_id = obj_id
            self.obj_name = 'corrupt'
            return
            
        self.cameras = cameras
        self.optimization_constraints = None
        self.train = False

        if not empty:
            # Initialize full object
            self.obj_id = obj_id
            self.scene_center = scene_center
            self.scene_points = scene_points
            self.scene_name = scene_name
            self.obj_name = obj_name
            self.constraints = []  # Spatial constraints (e.g., "on top of table")
            self.R = R  # Rotation matrix for alignment
            self.supported_objects = None  # Objects sitting on this object
            
            # Rendering/projection metadata
            self.xy_coords = None  # 2D pixel coordinates in rendered image
            self.projected_dominant_normal = None
            self.projected_points = None
            self.image_constraints = None  # Region in image (e.g., "top left")
            self.front_regions = None
            
            # Cached rendered views
            self.rendered_front = None
            self.front_inclined = None
            self.top_view_non_padded = None
            self.top_view_padded = None
            
            self.functionality = obj_name  # Functional description
            
            # Set all geometric properties
            self.set_obj_infos(np_array, set_dominant_normal_id=True, 
                             normals=normals, wall_objects=wall_objects)
            self.sort_surfaces()
        else:
            # Empty object for later loading
            self.scene_points = scene_points

    def load_from(self, obj_dict):
        """
        Load object from dictionary (deserialization).
        
        Args:
            obj_dict: Dictionary containing all object properties
        """
        if obj_dict['obj_name'] == 'corrupt':
            self.obj_id = obj_dict['obj_id']
            self.obj_name = obj_dict['obj_name']
            return
            
        # Load basic properties
        self.obj_id = obj_dict['obj_id']
        self.rendered_front = None
        self.front_inclined = None
        self.top_view_non_padded = None
        self.top_view_padded = None
        self.scene_center = obj_dict['scene_center']
        self.scene_name = obj_dict['scene_name']
        self.obj_name = obj_dict['obj_name']
        self.functionality = obj_dict['obj_name']
        self.constraints = obj_dict['constraints']
        
        # Load rendering metadata
        self.xy_coords = obj_dict['xy_coords']
        self.projected_points = obj_dict['projected_points']
        self.projected_dominant_normal = obj_dict['projected_dominant_normal']
        self.image_constraints = obj_dict['image_constraints']
        self.front_regions = obj_dict['front_regions']
        self.R = obj_dict['R']

        # Load point cloud data
        self.points_full = obj_dict['points_full']
        self.completed_points = obj_dict['completed_points']
        self.normals_full = obj_dict['normals_full']
        self.points = obj_dict['points']
        self.point_normals = obj_dict['point_normals']
        self.points_torch = torch.from_numpy(self.points)

        # Load bounding box properties
        self.volume = obj_dict['volume']
        self.corners_og = obj_dict['corners_og']
        self.corners_torch = obj_dict['corners_torch']
        self.faces = obj_dict['faces']
        self.corners = obj_dict['corners']
        self.min_point_torch = obj_dict['min_point_torch']
        self.min_points_torch = obj_dict['min_points_torch']
        
        # Load normals
        self.normals = obj_dict['normals']
        self.bx_normal_surfaces = obj_dict['bx_normal_surfaces']
        self.opposite_points = obj_dict['opposite_points']
        self.obb_corner_normals = obj_dict['obb_corner_normals']
        self.normals_torch = obj_dict['normals_torch']
        
        # Load dimensions
        self.width = obj_dict['width']
        self.height = obj_dict['height']
        self.depth = obj_dict['depth']
        self.center = obj_dict['center']
        self.center_torch = obj_dict['center_torch']

        # Load surfaces (support surfaces for placing objects)
        self.surfaces = obj_dict['surfaces']

        # Load dominant normal (front direction)
        self.dominant_normal_id = obj_dict['dominant_normal_id']
        self.dominant_normal = obj_dict['dominant_normal']
        self.dominant_normal_torch = obj_dict['dominant_normal_torch']
        self.dominant_normal_opposite_points = obj_dict['dominant_normal_opposite_points']
        
        # Load relationship data
        self.supported_objects = obj_dict['branches']
        self.sort_surfaces()

        # Reconstruct oriented bbox if corners available
        if self.corners_og is not None:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(self.corners_og)
            self.oriented_bbox = pc.get_minimal_oriented_bounding_box()
        else:
            self.oriented_bbox = None
    
    def construct_dictionary(self):
        """
        Serialize object to dictionary for saving.
        
        Returns:
            dict: Dictionary containing all object properties
        """
        if self.obj_name == 'corrupt':
            return {'obj_id': self.obj_id, 'obj_name': self.obj_name}
            
        obj_dict = {
            'obj_id': self.obj_id,
            'scene_center': self.scene_center,
            'scene_name': self.scene_name,
            'obj_name': self.obj_name,
            'constraints': self.constraints,
            'R': self.R,
            'projected_points': None,  # Too large to save
            'projected_dominant_normal': self.projected_dominant_normal,
            
            # Point cloud data
            'points_full': self.points_full,
            'completed_points': self.completed_points,
            'normals_full': self.normals_full,
            'points': self.points,
            'point_normals': self.point_normals,
            
            # Bounding box
            'volume': self.volume,
            'corners_og': self.corners_og,
            'corners_torch': self.corners_torch,
            'faces': self.faces,
            'corners': self.corners,
            'min_point_torch': self.min_point_torch,
            'min_points_torch': self.min_points_torch,
            
            # Normals
            'normals': self.normals,
            'bx_normal_surfaces': self.bx_normal_surfaces,
            'opposite_points': self.opposite_points,
            'obb_corner_normals': self.obb_corner_normals,
            'normals_torch': self.normals_torch,
            
            # Dimensions
            'width': self.width,
            'height': self.height,
            'depth': self.depth,
            'center': self.center,
            'center_torch': self.center_torch,
            
            # Surfaces
            'surfaces': self.surfaces,
            
            # Dominant normal
            'dominant_normal_id': self.dominant_normal_id,
            'dominant_normal': self.dominant_normal,
            'dominant_normal_torch': self.dominant_normal_torch,
            'dominant_normal_opposite_points': self.dominant_normal_opposite_points,
            
            # Relationships
            'branches': self.supported_objects,
            'xy_coords': self.xy_coords,
            'image_constraints': self.image_constraints,
            'front_regions': self.front_regions,
        }
        
        return obj_dict

    def set_obj_infos(self, np_array, normals, set_dominant_normal_id=False, wall_objects=None):
        """
        Initialize all geometric properties of the object.
        
        This method:
        1. Samples points for efficiency
        2. Computes oriented bounding box
        3. Extracts faces and normals
        4. Computes dimensions
        5. Detects support surfaces
        6. Determines dominant normal (front direction)
        
        Args:
            np_array: Full point cloud
            normals: Normal vectors
            set_dominant_normal_id: Whether to compute dominant normal
            wall_objects: Wall objects for context
        """
        # Sample points for efficiency (max 10,000 points)
        num_points = np_array.shape[0]
        num_sampled_points = max(1, int(0.2 * num_points))
        num_sampled_points = min(num_sampled_points, 10000)
        sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
        
        # Store full and sampled point clouds
        self.points_full = copy.deepcopy(np_array)
        self.completed_points = np.empty((0, 3))  # For future shape completion
        self.normals_full = copy.deepcopy(normals)
        self.points = np_array[sampled_indices]
        self.point_normals = (torch.from_numpy(normals[sampled_indices]) 
                             if normals is not None else None)
        self.points_torch = torch.from_numpy(self.points)
        
        # Create oriented bounding box
        # Use min/max z with xy points to get tight 2D bounds
        dummy_points = np.concatenate([
            np.concatenate([self.points[:, :2], 
                          np.ones((len(self.points), 1)) * self.points[:, 2].min()], axis=-1),
            np.concatenate([self.points[:, :2], 
                          np.ones((len(self.points), 1)) * self.points[:, 2].max()], axis=-1)
        ])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(dummy_points)
        self.oriented_bbox = point_cloud.get_minimal_oriented_bounding_box()
        
        # Calculate volume
        extents = self.oriented_bbox.extent
        self.volume = extents[0] * extents[1] * extents[2]
        
        # Extract corners and faces
        self.corners_og = np.asarray(self.oriented_bbox.get_box_points())
        self.corners_torch = torch.from_numpy(self.corners_og)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.corners_og)
        hull, _ = point_cloud.compute_convex_hull()
        self.faces = np.asarray(hull.triangles)
        self.corners = np.asarray(hull.vertices)
        
        # Store min points (for "on top of" constraints)
        self.min_point_torch = torch.from_numpy(np_array[np.argmin(np_array[:, -1])])
        min_z = np_array[:, -1].min()
        max_z = np_array[:, -1].max()
        self.min_points_torch = torch.from_numpy(
            np_array[np_array[:, -1] < min_z + (max_z - min_z) * 0.02]
        )[:, :2]
        
        # Compute face normals
        self.set_normals()
        
        # Set dimensions
        self.set_bbox_dimensions()
        
        # Detect support surfaces
        self.set_surfaces()
        
        # Compute orientation relative to scene
        self.set_object_scene_related_infos(self.scene_center, wall_objects=wall_objects)
        
        # Set dominant normal (front direction)
        self.set_dominant_normal()

    def object_orientation(self, trasn_z_angle=None):
        """
        Compute object's orientation angle in degrees.
        
        Returns angle of dominant normal relative to x-axis in XY plane.
        This represents which direction the object is "facing".
        
        Args:
            trasn_z_angle: Optional additional rotation to apply
            
        Returns:
            int: Orientation angle in degrees
        """
        dominant_normal = copy.deepcopy(self.dominant_normal)
        if trasn_z_angle is not None:
            R = rotation_matrix(np.array([0, 0, 1]), np.radians(trasn_z_angle))
            dominant_normal = R @ dominant_normal
            
        vector = dominant_normal
        magnitude = np.linalg.norm(vector)

        if magnitude == 0:
            return 0

        angle_x = np.arctan2(vector[1], vector[0]) * (180 / np.pi)
        return int(angle_x)
        
    def get_obj_details(self, get_position_and_orientation=False, get_surfaces=False,
                       trasn_z_angle=None, translate=None, in_obj_frame=False, add_angle=0):
        """
        Get object details dictionary.
        
        Args:
            get_position_and_orientation: Include position and orientation
            get_surfaces: Include surface information
            trasn_z_angle: Optional rotation angle
            translate: Optional translation vector
            in_obj_frame: Return in object's local frame
            add_angle: Additional angle to add
            
        Returns:
            dict: Object details including ID, name, dimensions, etc.
        """
        # Get predicted name (from annotations) or use obj_name
        pred_name = self.pred_obj_name[0] if self.pred_obj_name else self.obj_name
        if (self.obj_name != 'unknown') and ('cc' not in self.obj_name):
            pred_name = self.obj_name
        
        # Transform points if needed
        trasnformed_points = copy.deepcopy(self.points)
        if in_obj_frame:
            # Transform to object's local frame
            trasn_z_angle = -self.object_orientation() + add_angle
            max_point = trasnformed_points.max(axis=0)
            min_point = trasnformed_points.min(axis=0)
            dimentions = np.round(max_point - min_point, 2)
            center = np.round((max_point + min_point) / 2, 2)
            min_elev = trasnformed_points[:, 2].min()
            translate = -np.array([center[0], center[1], round(min_elev, 2)])
            
        if translate is not None:
            trasnformed_points += translate
        if trasn_z_angle is not None:
            R = rotation_matrix(np.array([0, 0, 1]), np.radians(trasn_z_angle))
            trasnformed_points = trasnformed_points @ R.T

        # Calculate dimensions
        max_point = trasnformed_points.max(axis=0)
        min_point = trasnformed_points.min(axis=0)
        dimentions = np.round(max_point - min_point, 2)
        center = np.round((max_point + min_point) / 2, 2)
        min_elev = trasnformed_points[:, 2].min()
        
        info = {
            'id': self.obj_id,
            'name': pred_name,
            'dimensions': dimentions.tolist(),
        }
        
        if get_position_and_orientation:
            info.update({
                'base': [center[0], center[1], round(min_elev, 2)],
                'oientation': self.object_orientation(trasn_z_angle),
            })
            
        if get_surfaces:
            info.update({
                'surfaces': self.get_surfaces_info(translate=translate)
            })
            
        return info
    
    def get_surfaces_info(self, translate=None):
        """Get information about support surfaces"""
        surfaces = {}
        for s_id, s in enumerate(self.surfaces):
            shift = 0 if translate is None else translate[-1]
            surfaces[s_id] = {
                'elevation': str(round(s['elevation'][-1].item() + shift, 2)) + 'meters'
            }
        return surfaces

    def get_base_points(self, perc=0.05):
        """
        Get points at the base of the object (bottom 5%).
        
        Used for determining placement and "on top of" relationships.
        
        Args:
            perc: Percentage of height to consider as base
            
        Returns:
            Tensor of XY coordinates at object base
        """
        min_point = torch.min(self.points_torch[:, 2])
        max_point = torch.max(self.points_torch[:, 2])
        upper_bound = min_point + perc * (max_point - min_point)
        return self.points_torch[self.points_torch[:, 2] < upper_bound][:, :2]
        
    def sample_free_point_from_surface(self, number_points, region_name=None, surface_id=None):
        """
        Sample free points from a surface for object placement.
        
        Args:
            number_points: Number of points to sample
            region_name: Specific region on surface (e.g., "center", "left")
            surface_id: Specific surface ID
            
        Returns:
            Generator yielding sampled points
        """
        if len(self.surfaces) == 0 or number_points == 0:
            return torch.empty((0, 3))
            
        # Select surface (random if not specified)
        surface = (self.surfaces[surface_id] if surface_id is not None 
                  else random.choices(self.surfaces, [s['prob'] for s in self.surfaces], k=1)[0])
        
        # Select region (random if not specified)
        region = (surface['regions'][region_name] if region_name is not None 
                 else random.choices(list(surface['regions'].values()), 
                                   surface['regions probs'], k=1)[0])
        
        # Sample points using farthest point sampling
        points = region[torch.from_numpy(
            fpsample.fps_sampling(region.numpy(), number_points).astype(int)
        )]

        # Return generator that cycles through points
        def point_generator():
            i = 0
            while True:
                yield points[i]
                i = (i + 1) % len(points)
                
        return point_generator()

    def set_constraints(self, constraints):
        """Set spatial constraints (e.g., ['on top of', 'table', 5])"""
        self.constraints = constraints

    def get_intersection(self, obj, add_height):
        """
        Calculate intersection volume ratio with another object.
        
        Used for determining spatial relationships and collisions.
        
        Args:
            obj: Other object
            add_height: Additional height to add to other object's bbox
            
        Returns:
            float: Ratio of intersection volume to this object's volume
        """
        if obj.obj_name == 'corrupt' or self.obj_name == 'corrupt':
            return 0.
            
        # Convert oriented bboxes to axis-aligned
        aabb1 = self.oriented_bbox.get_axis_aligned_bounding_box()
        aabb2 = obj.oriented_bbox.get_axis_aligned_bounding_box()

        min1 = np.array(aabb1.min_bound)
        max1 = np.array(aabb1.max_bound)
        min2 = np.array(aabb2.min_bound)
        max2 = np.array(aabb2.max_bound)
        
        # Shrink other object slightly in XY
        center = (min2 + max2) / 2
        min2[:2] = (min2[:2] - center[:2]) * 0.9 + center[:2]
        max2[:2] = (max2[:2] - center[:2]) * 0.9 + center[:2]
        max2[2] += add_height
        
        # Calculate intersection
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        if np.all(intersection_min < intersection_max):
            intersection_volume = np.prod(intersection_max - intersection_min)
            return intersection_volume / np.prod(max1 - min1)
        else:
            return 0.0

    def set_normals(self):
        """
        Compute face normals for bounding box faces.
        
        For each face:
        - Calculate normal via cross product
        - Find opposite face (parallel face on other side)
        - Merge similar faces (dot product > 0.95)
        - Store areas for dominant face detection
        """
        self.normals = []
        self.bx_normal_surfaces = []  # Face areas
        self.faces_merged = []
        self.opposite_points = []
        self.obb_corner_normals = [[] for _ in range(8)]
        
        # Calculate normal for each face
        for i, face in enumerate(self.faces):
            p1, p2, p3 = self.corners[face[0]], self.corners[face[1]], self.corners[face[2]]
            
            edge1 = p2 - p1
            edge2 = p3 - p1
            
            normal = np.cross(edge1, edge2)
            normal /= np.linalg.norm(normal)
            
            # Store normal for each corner
            for corner_idx in range(3):
                self.obb_corner_normals[face[corner_idx]].append(normal)
                
            self.normals.append(normal)
            area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
            self.bx_normal_surfaces.append(area)
            
            # Find opposite face (parallel, opposite direction)
            for j, other_face in enumerate(self.faces):
                if i != j:
                    op1, op2, op3 = self.corners[other_face[0]], self.corners[other_face[1]], self.corners[other_face[2]]
                    edge_op1 = op2 - op1
                    edge_op2 = op3 - op1
                    other_normal = np.cross(edge_op1, edge_op2)
                    other_normal /= np.linalg.norm(other_normal)
                    
                    dot_product = np.dot(normal, other_normal)
                    
                    # Opposite face has dot product < -0.9
                    if dot_product < -0.9:
                        opposite_face_index = j
                        break
            
            # Store opposite face points
            if opposite_face_index is not None:
                opp_face = self.faces[opposite_face_index]
                self.opposite_points.append(np.stack([self.corners[idx] for idx in opp_face]))
            else:
                self.opposite_points.append(None)
        
        # Average normals for each corner
        self.obb_corner_normals = np.stack([
            np.stack(i).mean(0) for i in self.obb_corner_normals
        ])
        self.normals = np.stack(self.normals)
        self.bx_normal_surfaces = np.array(self.bx_normal_surfaces)

        # Merge similar faces (dot product > 0.95)
        self.new_normals = []
        self.new_bx_normal_surfaces = []
        used_faces = set()
        self.opposite_points_new = []
        
        for i, face in enumerate(self.faces):
            if i in used_faces:
                continue

            merged_face = list(face)
            current_normal = self.normals[i]

            # Find similar faces to merge
            for j, other_face in enumerate(self.faces):
                if i != j and j not in used_faces:
                    other_normal = self.normals[j]
                    dot_product = np.dot(current_normal, other_normal)

                    if dot_product > 0.95:
                        merged_face += [idx for idx in other_face if idx not in merged_face]
                        used_faces.add(j)
                        break

            # Pad to 4 indices
            while len(merged_face) < 4:
                merged_face.append(None)
            merged_face = merged_face[:4]

            self.faces_merged.append(merged_face)
            self.new_normals.append(current_normal)
            self.new_bx_normal_surfaces.append(self.bx_normal_surfaces[i] * 2)
            used_faces.add(i)
            self.opposite_points_new.append(self.opposite_points[i])

        # Update with merged faces
        self.faces = np.array(self.faces_merged).astype(int)
        self.normals = np.stack(self.new_normals)
        self.normals_torch = torch.from_numpy(self.normals)
        self.opposite_points = np.stack(self.opposite_points_new)
        self.bx_normal_surfaces = np.array(self.new_bx_normal_surfaces)

    def set_bbox_dimensions(self):
        """Calculate width, height, depth from oriented bounding box"""
        bbox_min = self.oriented_bbox.get_min_bound()
        bbox_max = self.oriented_bbox.get_max_bound()
        
        self.width = bbox_max[0] - bbox_min[0]
        self.height = bbox_max[2] - bbox_min[2]
        self.depth = bbox_max[1] - bbox_min[1]
        self.center = (bbox_min + bbox_max) / 2.0
        self.center_torch = torch.from_numpy(self.center)
    
    def compute_iou(self, other):
        """
        Compute 3D IoU with another object using PyTorch3D.
        
        Args:
            other: Other OBJ_POINT_CLOUD object
            
        Returns:
            float: 3D Intersection over Union
        """
        _, iou_3d = box3d_overlap(
            self.corners_torch[None, ...].float(),
            other.corners_torch[None, ...].float()
        )
        return iou_3d.item()
    
    def reset_dominant_normal(self):
        """
        Recompute the dominant normal (front direction) of the object.
        
        This is a critical method that determines which face of the bounding box
        should be considered the "front" of the object. The algorithm:
        
        1. For objects with clear directionality (not floor):
           - Find non-vertical faces
           - Project points onto each face plane
           - Choose face with most points/best symmetry
           - Prefer face farthest from object center
        
        2. For special cases (floor, furniture):
           - Use mean normal of point cloud
           - Consider scene context
           - Use wall alignment for tables
        
        This affects:
        - Object orientation computation
        - Rendering front views
        - Spatial relationship understanding
        """
        if self.obj_name == 'corrupt':
            return
            
        z_axis = np.array([0, 0, 1])
        
        # Get normals in XY plane (not vertical)
        if self.obj_name != 'floor':
            xy_oriented_normals = self.point_normals[
                np.abs(self.point_normals @ z_axis) < 0.02
            ]
        else:
            xy_oriented_normals = self.surfaces[0]['contour normals'].numpy()
            
        xy_oriented_normals /= np.linalg.norm(xy_oriented_normals, axis=-1, keepdims=True)
        
        # Check if object has clear directionality
        has_clear_direction = (
            (np.linalg.norm(xy_oriented_normals.mean(axis=0)) > 0.1 or 
             self.obj_name == 'floor') and
            (self.obj_name not in ['chair', 'sofa', 'couch', 'armchair'])
        )
        
        if has_clear_direction:
            # Use mean normal direction
            mean_normal = self.point_normals.mean(axis=0)
            dot_products_with_z = np.abs(self.normals @ z_axis)
            alignment_threshold = 0.9
            non_z_aligned_indices = np.where(dot_products_with_z < alignment_threshold)[0]
            filtered_normals = self.normals[non_z_aligned_indices]
            
            if self.obj_name != 'floor':
                # Count points visible from each face
                num_points = []
                for n in filtered_normals:
                    projected_points = np.unique(
                        project_point_cloud_to_plane(self.points, n) // 0.05, axis=0
                    )
                    num_points.append(len(projected_points))
                    
                # Choose top 2 candidates
                normal_candidates = np.argsort(np.array(num_points))[-2:]
                
                # Pick one most aligned with mean normal
                sim_to_mean_norm = [np.dot(mean_normal, n) for n in filtered_normals[normal_candidates]]
                self.dominant_normal_id = non_z_aligned_indices[
                    normal_candidates[sim_to_mean_norm.index(max(sim_to_mean_norm))]
                ]
            else:
                # For floor, just use most aligned with mean
                sim_to_mean_norm = [np.dot(mean_normal, n) for n in filtered_normals]
                self.dominant_normal_id = non_z_aligned_indices[sim_to_mean_norm.index(max(sim_to_mean_norm))]
                
            self.set_dominant_normal()
            self.scene_points = None
            return

        # For objects without clear directionality (e.g., chairs, tables)
        dot_products_with_z = np.abs(self.normals @ z_axis)
        alignment_threshold = 0.9
        non_z_aligned_indices = np.where(dot_products_with_z < alignment_threshold)[0]
        filtered_normals = self.normals[non_z_aligned_indices]
        
        # Project onto each face and count points
        num_points = []
        for n in filtered_normals:
            projected_points = np.unique(
                project_point_cloud_to_plane(self.points, n) // 0.05, axis=0
            )
            num_points.append(len(projected_points))
            
        normal_candidates = np.argsort(np.array(num_points))[-2:]
        self.set_dominant_normal()
        
        # Use scene context if available
        max_point = self.corners.max(axis=0)
        min_point = self.corners.min(axis=0)
        
        if self.surfaces and (np.abs(max_point[-1] - self.surfaces[0]['elevation'][-1]) < 0.06) and (self.scene_points is not None):
            # Object has surface at top - use scene points above it
            min_point[-1] = self.surfaces[0]['elevation'][-1] - 0.09
            bbox = np.stack([min_point, max_point])
            bbox[1, -1] += (max_point[-1] - min_point[-1])
            
            points_in_box = self.scene_points[
                (self.scene_points[:, 0] > bbox[0, 0]) & (self.scene_points[:, 0] < bbox[1, 0]) &
                (self.scene_points[:, 1] > bbox[0, 1]) & (self.scene_points[:, 1] < bbox[1, 1]) &
                (self.scene_points[:, 2] > bbox[0, 2]) & (self.scene_points[:, 2] < bbox[1, 2])
            ]
            
            # Filter points inside surface polygon
            b_size = 5000
            points_in_polygon = []
            for b in range(0, len(points_in_box), b_size):
                points_in_box_batch = points_in_box[b : b + b_size]
                points_in_polygon_batch = points_in_box_batch[
                    inside_polygon_loss(
                        self.surfaces[0]["contour points"].cuda(),
                        self.surfaces[0]["contour normals"].cuda(),
                        torch.from_numpy(points_in_box_batch[:, :2]).cuda(),
                        return_mean=False
                    ).cpu().numpy() == 0
                ]
                points_in_polygon.append(points_in_polygon_batch)
                
            points_scene = np.concatenate(points_in_polygon + [self.points])
        else:
            points_scene = self.points
            
        # Find region with most points at top
        ref_point = find_max_adjacent_regions(points_scene)
        
        # Choose face farthest from reference point
        distances_to_centroid = np.stack([
            np.linalg.norm(self.corners[self.faces[idx]].mean(axis=0)[:2] - ref_point[:2])
            for idx in non_z_aligned_indices[normal_candidates]
        ])
        self.dominant_normal_id = non_z_aligned_indices[
            normal_candidates[np.argmax(distances_to_centroid)]
        ]
        self.set_dominant_normal()
        self.scene_points = None

    def set_object_scene_related_infos(self, scene_center, wall_objects=None):
        """
        Compute object properties relative to scene.
        
        Used during initialization to determine dominant normal based on:
        - Scene center position
        - Wall positions (for tables/desks)
        - Objects above (for surfaces)
        
        Args:
            scene_center: Center point of entire scene
            wall_objects: List of wall objects for context
        """
        mean_normal = self.point_normals.mean(axis=0)
        z_axis = np.array([0, 0, 1])
        dot_products_with_z = np.abs(self.normals @ z_axis)
        alignment_threshold = 0.9
        non_z_aligned_indices = np.where(dot_products_with_z < alignment_threshold)[0]
        filtered_normals = self.normals[non_z_aligned_indices]
        
        # Find most and least aligned with mean normal
        sim_to_mean_normal = [np.dot(mean_normal, n) for n in filtered_normals]
        self.dominant_normal_id = non_z_aligned_indices[sim_to_mean_normal.index(max(sim_to_mean_normal))]
        opposite_normal_id = non_z_aligned_indices[sim_to_mean_normal.index(min(sim_to_mean_normal))]
        normal_candidates = [self.dominant_normal_id, opposite_normal_id]
        
        self.set_dominant_normal()
        
        # Use scene context to choose between candidates
        max_point = self.corners.max(axis=0)
        min_point = self.corners.min(axis=0)
        
        if self.surfaces and (np.abs(max_point[-1] - self.surfaces[0]['elevation'][-1]) < 0.06) and (self.scene_points is not None):
            # Similar logic to reset_dominant_normal
            min_point[-1] = self.surfaces[0]['elevation'][-1] - 0.09
            bbox = np.stack([min_point, max_point])
            bbox[1, -1] += (max_point[-1] - min_point[-1])
            
            points_in_box = self.scene_points[
                (self.scene_points[:, 0] > bbox[0, 0]) & (self.scene_points[:, 0] < bbox[1, 0]) &
                (self.scene_points[:, 1] > bbox[0, 1]) & (self.scene_points[:, 1] < bbox[1, 1]) &
                (self.scene_points[:, 2] > bbox[0, 2]) & (self.scene_points[:, 2] < bbox[1, 2])
            ]
            
            b_size = 5000
            points_in_polygon = []
            for b in range(0, len(points_in_box), b_size):
                points_in_box_batch = points_in_box[b : b + b_size]
                points_in_polygon_batch = points_in_box_batch[
                    inside_polygon_loss(
                        self.surfaces[0]["contour points"].cuda(),
                        self.surfaces[0]["contour normals"].cuda(),
                        torch.from_numpy(points_in_box_batch[:, :2]).cuda(),
                        return_mean=False
                    ).cpu().numpy() == 0
                ]
                points_in_polygon.append(points_in_polygon_batch)
                
            points_scene = np.concatenate(points_in_polygon + [self.points])
        else:
            points_scene = self.points
            
        # Use top points as reference
        minimum_allowed = points_scene[:, 2].min() + 0.95 * (points_scene[:, 2].max() - points_scene[:, 2].min())
        top_75p_points = points_scene[points_scene[:, 2] > minimum_allowed]
        ref_point = top_75p_points.mean(axis=0)
        
        # Choose face farthest from reference
        distances_to_centroid = np.stack([
            np.linalg.norm(self.corners[self.faces[idx]].mean(axis=0)[:2] - ref_point[:2])
            for idx in normal_candidates
        ])
        self.dominant_normal_id = normal_candidates[np.argmax(distances_to_centroid)]

    def object_in_surface(self, t_self, obj, t_obj):
        """
        Check if another object is inside one of this object's surfaces.
        
        Used for determining "inside" relationships (e.g., book inside shelf).
        
        Args:
            t_self: Transformation matrix for this object
            obj: Other object to check
            t_obj: Transformation matrix for other object
            
        Returns:
            Tuple of (is_inside, surface_id, object_id)
        """
        # Transform other object's points
        points_homogonous = np.hstack((obj.points, np.ones((obj.points.shape[0], 1))))
        points = (points_homogonous @ t_obj.T)[:, :3]
        min_point = points[:, 2].min()
        
        # Check each surface
        for s_id, s in enumerate(self.surfaces):
            # Transform surface
            elevation_homogeneous = np.hstack((s['elevation'], [1]))
            surface_elevation = (t_self @ elevation_homogeneous)[:3]
            contour_points_homogeneous = np.hstack((
                s['contour points'], np.ones((s['contour points'].shape[0], 1))
            ))
            polygon_points = (contour_points_homogeneous @ t_self.T)[:, :3]

            surface_elevation = surface_elevation[-1].item()
            polygon_normals = s['contour normals'] @ t_self[:3, :3].T
            
            # Check if points are inside polygon and at right height
            loss = inside_polygon_loss(
                polygon_points.cuda(), polygon_normals.cuda(),
                torch.from_numpy(points).cuda()
            )
            if (loss < 0.5) and (abs(surface_elevation - min_point) < 0.1):
                return True, s_id, self.obj_id
        
        return False, None, self.obj_id
    
    def set_dominant_normal(self):
        """Set dominant normal vector and opposite points"""
        self.dominant_normal = self.normals[self.dominant_normal_id]
        self.dominant_normal_torch = torch.from_numpy(self.dominant_normal)
        
        if self.dominant_normal_id is not None:
            opposite_points = self.opposite_points[self.dominant_normal_id]
            self.dominant_normal_opposite_points = torch.from_numpy(opposite_points)
        else:
            self.dominant_normal_opposite_points = None
    
    def set_surfaces(self, bin_width=0.1, grid_resolution=0.01, plot_contours=False,
                    plot_surfaces=False, debug=False, projected_supported_objects=None):
        """
        Detect horizontal support surfaces on the object.
        
        Surfaces are regions where other objects can be placed (e.g., tabletop,
        shelves). The algorithm:
        
        1. Bin points by Z height
        2. For bins with >500 points, create 2D occupancy grid
        3. Find contour of occupied region
        4. Smooth contour with splines
        5. Calculate normals pointing inward
        6. Divide into 9 regions (3x3 grid) for placement
        7. Sort by elevation and area
        
        Args:
            bin_width: Height of each bin (meters)
            grid_resolution: Size of grid cells (meters)
            plot_contours: Whether to visualize contours
            plot_surfaces: Whether to visualize surfaces
            debug: Enable debug output
            projected_supported_objects: Optional points of objects on surface
        """
        if self.R is not None:
            R = self.R
        else:
            R = np.eye(3)
            
        # Merge full points with any supported object points
        if projected_supported_objects is None:
            merged_points = np.concatenate([self.points_full @ R.T, self.completed_points], axis=0)
        else:
            merged_points = np.concatenate(
                [s['free points'].numpy() for s in self.surfaces] + [projected_supported_objects]
            )
            
        z_values = merged_points[:, 2]
        z_min, z_max = np.min(z_values), np.max(z_values)

        # Create bins
        bins = np.arange(z_min - bin_width, z_max + bin_width, bin_width)
        counts, bin_edges = np.histogram(z_values, bins=bins)

        surfaces_current = []
        surfaces_areas = []

        # Process each bin
        for i in range(len(counts)):
            if counts[i] > 500:  # Minimum points for valid surface
                bin_mask = (z_values >= bin_edges[i]) & (z_values <= bin_edges[i + 1])
                bin_mask_og = (z_values >= bin_edges[i]) & (z_values <= bin_edges[i + 1])
                surface_z_value = z_values[bin_mask_og].mean()
                
                points_in_bin = merged_points[bin_mask]
                
                # Extend grid with nearby points
                extended_grid_points = np.concatenate([
                    points_in_bin[:, :2],
                    merged_points[:, :2][
                        (merged_points[:, -1] > surface_z_value - self.height / 2) &
                        (merged_points[:, -1] < surface_z_value + self.height / 2)
                    ]
                ])
                
                min_xy = extended_grid_points.min(axis=0)[:2]

                # Create occupancy grid
                grid_xy = np.unique(((extended_grid_points - min_xy) // grid_resolution).astype(int), axis=0)
                max_xy = grid_xy.max(axis=0)
                grid = np.zeros((max_xy[0] + 1, max_xy[1] + 1), dtype=bool)
                grid[grid_xy[:, 0], grid_xy[:, 1]] = True

                # Smooth grid
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
                if len(contour_points) < 4:
                    continue

                # Add interpolation points to close contour
                num_points = 4
                contour_points = np.concatenate([
                    contour_points,
                    np.stack([
                        contour_points[-1] + ((i + 1) / (num_points + 1)) * (contour_points[0] - contour_points[-1])
                        for i in range(num_points)
                    ])
                ])
                
                # Create free points (points on surface)
                free_points = torch.from_numpy(np.concatenate([
                    points_in_bin[:, :2],
                    (np.ones((points_in_bin.shape[0], 1)) * surface_z_value)
                ], axis=-1))
                
                # Divide surface into 9 regions if cameras available
                regions = {}
                if self.cameras is not None:
                    projected_free_points = self.cameras.transform_points_screen(
                        free_points.cuda().float()
                    )[:, :2].cpu()
                    max_end = projected_free_points.max(dim=0).values
                    min_end = projected_free_points.min(dim=0).values
                    surface_2d_W, surface_2d_H = (max_end - min_end).cpu().tolist()
                    height_third = surface_2d_H // 3
                    width_third = surface_2d_W // 3
                    projected_free_points -= min_end
                    
                    # Define 9 regions
                    regions["top left"] = free_points[
                        ((projected_free_points[:, 0] < width_third) & 
                         (projected_free_points[:, 1] < height_third)).bool()
                    ]
                    regions["top"] = free_points[
                        ((projected_free_points[:, 0] >= width_third) & 
                         (projected_free_points[:, 0] < 2 * width_third) & 
                         (projected_free_points[:, 1] < height_third)).bool()
                    ]
                    regions["top right"] = free_points[
                        ((projected_free_points[:, 0] >= 2 * width_third) & 
                         (projected_free_points[:, 1] < height_third)).bool()
                    ]
                    regions["left"] = free_points[
                        ((projected_free_points[:, 0] < width_third) & 
                         (projected_free_points[:, 1] >= height_third) & 
                         (projected_free_points[:, 1] < 2 * height_third)).bool()
                    ]
                    regions["center"] = free_points[
                        ((projected_free_points[:, 0] >= width_third) & 
                         (projected_free_points[:, 0] < 2 * width_third) & 
                         (projected_free_points[:, 1] >= height_third) & 
                         (projected_free_points[:, 1] < 2 * height_third)).bool()
                    ]
                    regions["right"] = free_points[
                        ((projected_free_points[:, 0] >= 2 * width_third) & 
                         (projected_free_points[:, 1] >= height_third) & 
                         (projected_free_points[:, 1] < 2 * height_third)).bool()
                    ]
                    regions["bottom left"] = free_points[
                        ((projected_free_points[:, 0] < width_third) & 
                         (projected_free_points[:, 1] >= 2 * height_third)).bool()
                    ]
                    regions["bottom"] = free_points[
                        ((projected_free_points[:, 0] >= width_third) & 
                         (projected_free_points[:, 0] < 2 * width_third) & 
                         (projected_free_points[:, 1] >= 2 * height_third)).bool()
                    ]
                    regions["bottom right"] = free_points[
                        ((projected_free_points[:, 0] >= 2 * width_third) & 
                         (projected_free_points[:, 1] >= 2 * height_third)).bool()
                    ]

                regions_probs = [
                    len(v) / len(free_points) if len(v) != 0 else 0.0
                    for k, v in regions.items()
                ]

                # Smooth contour with splines
                if len(contour_points) > 4 + num_points:
                    try:
                        tck, u = splprep(contour_points.T, s=0)
                        contour_points = np.array(splev(u, tck)).T
                    except:
                        try:
                            tck, u = splprep(contour_points[::5].T, s=0)
                            contour_points = np.array(splev(u, tck)).T
                        except:
                            continue

                    # Calculate normals
                    tangents = np.gradient(contour_points, axis=0)
                    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
                    normals = np.empty_like(tangents)
                    normals[:, 0] = -tangents[:, 1]
                    normals[:, 1] = tangents[:, 0]
                    
                    # Calculate area
                    area = len(torch.unique(free_points[:, :2] // 0.02, dim=0))
                    surfaces_areas.append(area)
                    
                    # Store surface
                    surfaces_current.append({
                        'elevation': torch.tensor([0.0, 0.0, surface_z_value]) @ torch.from_numpy(R),
                        'free points': free_points @ torch.from_numpy(R),
                        'regions': regions,
                        'regions probs': regions_probs,
                        'contour points': torch.from_numpy(np.concatenate([
                            contour_points,
                            (np.ones((contour_points.shape[0], 1)) * surface_z_value)
                        ], axis=-1) @ R),
                        'contour normals': torch.from_numpy(np.concatenate([
                            -normals,
                            (np.zeros((normals.shape[0], 1)))
                        ], axis=-1) @ R),
                        'contour center': torch.from_numpy(contour_points.mean(0) @ R[:2, :2]),
                        'occ grid': {
                            'coords': torch.from_numpy(np.stack(np.where(grid)).T),
                            'min': torch.from_numpy(min_xy),
                            'res': grid_resolution
                        },
                        'area': area,
                        'R': R
                    })

        # Filter and sort surfaces
        surfaces = []
        if len(surfaces_current) > 0:
            sum_area = max(surfaces_areas)
            probs = [count / sum_area for count in surfaces_areas]
            probs_new = []
            filtered_surfaces = []

            # Filter small surfaces
            for s_i, surface in enumerate(surfaces_current):
                if probs[s_i] < 0.15:
                    continue
                probs_new.append(probs[s_i])
                filtered_surfaces.append(surface)

            # Normalize probabilities
            probs_new = [p / sum(probs_new) for p in probs_new]

            # Assign probabilities
            for p, s in zip(probs_new, filtered_surfaces):
                s['prob'] = p

            # Sort by probability
            sorted_surfaces = sorted(filtered_surfaces, key=lambda x: x['prob'], reverse=True)
            probs_new = [s['prob'] for s in sorted_surfaces]
            surfaces = sorted_surfaces

            # Remove overlapping surfaces at same height
            surface_elevation = np.array([s['elevation'][-1].item() for s in surfaces])
            p_max = max(probs_new)
            hieght_h_p = surface_elevation[probs_new.index(p_max)]
            remove_ids = []
            for s_id, (p, h) in enumerate(zip(probs_new, surface_elevation)):
                if h > hieght_h_p - 0.05 and p < 0.8 * p_max:
                    remove_ids.append(s_id)
            surfaces = [s for s_id, s in enumerate(surfaces) if s_id not in remove_ids]
            
            # Keep only surfaces with significant height difference
            surface_elevation = np.array([s['elevation'][-1].item() for s in surfaces])
            surface_elevation_ids_sorted = np.argsort(surface_elevation)[::-1]

            sorted_elevations = surface_elevation[surface_elevation_ids_sorted]

            keep_id = [surface_elevation_ids_sorted[0]] + surface_elevation_ids_sorted[
                np.where((sorted_elevations[:-1] - sorted_elevations[1:]) > 0.2)[0] + 1
            ].tolist()
            surfaces = [surfaces[idx] for idx in keep_id]

        # Visualize if requested
        if plot_contours:
            for surface in surfaces:
                contour_points = surface['contour points']
                free_points = surface['free points']
                contour_normals = surface['contour normals']

                plt.figure(figsize=(8, 8))
                plt.plot(contour_points[:, 0], contour_points[:, 1], 'o-', label='Polygon')

                for i, point in enumerate(contour_points[:-1]):
                    plt.arrow(point[0], point[1], contour_normals[i][0] * 0.1, 
                            contour_normals[i][1] * 0.1, head_width=0.02, head_length=0.02, 
                            fc='r', ec='r')

                plt.scatter(contour_points[:, 0], contour_points[:, 1], c='blue', label='Contour Points')
                plt.scatter(free_points[:, 0], free_points[:, 1], label='free points', c='red')

                plt.xlabel("X-axis")
                plt.ylabel("Y-axis")
                plt.legend()
                plt.title(f"{self.obj_name}: Surface at Elevation {surface['elevation'][-1].item():.2f}")
                plt.grid(True)
                plt.axis('equal')
                plt.show()

        self.surfaces = surfaces
        
        # Clean up temporary data
        if projected_supported_objects is not None:
            self.points_full = None
            self.normals_full = None
        self.completed_points = None

    def get_2d_bbox(self, cameras):
        """
        Get 2D bounding box in image coordinates.
        
        Args:
            cameras: Tuple of (camera, min_point_cropped)
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        camera = cameras[0]
        min_point_cropped = cameras[1]
        projected_points = camera.transform_points_screen(
            self.points_torch.cuda().float()
        ).cpu().numpy()
        min_point = projected_points[:, :2].min(axis=0) - min_point_cropped
        max_point = projected_points[:, :2].max(axis=0) - min_point_cropped
        return (int(min_point[0]), int(min_point[1]), int(max_point[0]), int(max_point[1]))
        
    def update_object_info(self, center, t):
        """
        Update object after transformation.
        
        This method applies a transformation matrix to update all geometric
        properties of the object including:
        - Point cloud positions
        - Bounding box corners
        - Normal vectors
        - Surface contours and elevations
        - Region divisions
        
        Args:
            center: Center of rotation (numpy array [3])
            t: 4x4 transformation matrix
        """
        # Create transformation relative to center
        translation_to_origin = np.eye(4)
        translation_to_origin[:3, 3] = -center

        transformation_at_origin = t @ translation_to_origin

        translation_back_to_center = np.eye(4)
        translation_back_to_center[:3, 3] = center

        transformation = translation_back_to_center @ transformation_at_origin

        # Update points
        points_homogeneous = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        self.points = (points_homogeneous @ transformation.T)[:, :3]
        self.points_torch = torch.from_numpy(self.points)
        self.point_normals = self.point_normals @ torch.from_numpy(t[:3, :3]).T

        # Update bounding box corners
        corners_homogeneous = np.hstack((self.corners_og, np.ones((self.corners_og.shape[0], 1))))
        self.corners_og = (corners_homogeneous @ transformation.T)[:, :3]
        self.corners_torch = torch.from_numpy(self.corners_og)

        # Update normals
        self.normals = self.normals @ t[:3, :3].T
        self.normals_torch = torch.from_numpy(self.normals)

        corners_homogeneous = np.hstack((self.corners, np.ones((self.corners.shape[0], 1))))
        self.corners = (corners_homogeneous @ transformation.T)[:, :3]

        # Update opposite points
        opposite_points_homogenous = np.concatenate([
            self.opposite_points,
            np.ones((self.opposite_points.shape[0], self.opposite_points.shape[1], 1))
        ], axis=-1)
        self.opposite_points = (opposite_points_homogenous @ transformation.T)[..., :3]
        
        # Update each surface
        for s in self.surfaces:
            # Update elevation
            elevation_homogeneous = np.hstack((s['elevation'], [1]))
            s['elevation'] = (transformation @ elevation_homogeneous)[:3]

            # Update free points
            free_points_homogeneous = torch.cat([
                s['free points'],
                torch.ones((s['free points'].shape[0], 1))
            ], dim=-1)
            s['free points'] = (free_points_homogeneous @ torch.from_numpy(transformation).T)[:, :3]

            # Update contour points
            contour_points_homogeneous = np.hstack((
                s['contour points'],
                np.ones((s['contour points'].shape[0], 1))
            ))
            s['contour points'] = (contour_points_homogeneous @ transformation.T)[:, :3]

            # Update contour normals
            s['contour normals'] = s['contour normals'] @ t[:3, :3].T

            # Update contour center
            contour_center_homogeneous = np.hstack((s['contour center'], [1], [1]))
            s['contour center'] = (transformation @ contour_center_homogeneous)[:2]

            # Update regions if cameras available
            regions = {}
            if self.cameras is not None:
                projected_free_points = self.cameras.transform_points_screen(
                    s['free points'].cuda().float()
                )[:, :2].cpu()
                max_end = projected_free_points.max(dim=0).values
                min_end = projected_free_points.min(dim=0).values
                surface_2d_W, surface_2d_H = (max_end - min_end).cpu().tolist()
                height_third = surface_2d_H // 3
                width_third = surface_2d_W // 3
                projected_free_points -= min_end
                
                # Redefine 9 regions after transformation
                regions["top left"] = s['free points'][
                    ((projected_free_points[:, 0] < width_third) & 
                     (projected_free_points[:, 1] < height_third)).bool()
                ]
                regions["top"] = s['free points'][
                    ((projected_free_points[:, 0] >= width_third) & 
                     (projected_free_points[:, 0] < 2 * width_third) & 
                     (projected_free_points[:, 1] < height_third)).bool()
                ]
                regions["top right"] = s['free points'][
                    ((projected_free_points[:, 0] >= 2 * width_third) & 
                     (projected_free_points[:, 1] < height_third)).bool()
                ]
                regions["left"] = s['free points'][
                    ((projected_free_points[:, 0] < width_third) & 
                     (projected_free_points[:, 1] >= height_third) & 
                     (projected_free_points[:, 1] < 2 * height_third)).bool()
                ]
                regions["center"] = s['free points'][
                    ((projected_free_points[:, 0] >= width_third) & 
                     (projected_free_points[:, 0] < 2 * width_third) & 
                     (projected_free_points[:, 1] >= height_third) & 
                     (projected_free_points[:, 1] < 2 * height_third)).bool()
                ]
                regions["right"] = s['free points'][
                    ((projected_free_points[:, 0] >= 2 * width_third) & 
                     (projected_free_points[:, 1] >= height_third) & 
                     (projected_free_points[:, 1] < 2 * height_third)).bool()
                ]
                regions["bottom left"] = s['free points'][
                    ((projected_free_points[:, 0] < width_third) & 
                     (projected_free_points[:, 1] >= 2 * height_third)).bool()
                ]
                regions["bottom"] = s['free points'][
                    ((projected_free_points[:, 0] >= width_third) & 
                     (projected_free_points[:, 0] < 2 * width_third) & 
                     (projected_free_points[:, 1] >= 2 * height_third)).bool()
                ]
                regions["bottom right"] = s['free points'][
                    ((projected_free_points[:, 0] >= 2 * width_third) & 
                     (projected_free_points[:, 1] >= 2 * height_third)).bool()
                ]
            
            # Update region probabilities
            regions_probs = [
                len(v) / len(s['free points']) if len(v) != 0 else 0.0
                for k, v in regions.items()
            ]
            s['regions'] = regions
            s['regions probs'] = regions_probs
            
        # Update dominant normal
        self.set_dominant_normal()
        
    def sort_surfaces(self):
        """Sort surfaces by elevation (highest first)"""
        surface_elevations = np.array([
            s['elevation'][-1].item() for s_id, s in enumerate(self.surfaces)
        ])
        sorted_ids = np.argsort(surface_elevations)[::-1]
        self.surfaces = [self.surfaces[id] for id in sorted_ids]

    def visualize_multiple(self, objs, vizualize_surfaces=False, return_geometries=False):
        """
        Visualize this object along with other objects.
        
        Args:
            objs: List of other OBJ_POINT_CLOUD objects
            vizualize_surfaces: Whether to show surface normals
            return_geometries: If True, return geometries instead of displaying
            
        Returns:
            List of Open3D geometries if return_geometries=True
        """
        pcs = []
        colors = generate_vibrant_cool_colors(len(objs))
        pcs.extend(self.visualize(return_pc=True, vizualize_surfaces=vizualize_surfaces))
        for color, obj in zip(colors, objs):
            pcs.extend(obj.visualize(return_pc=True, obb_color=color))
        if return_geometries:
            return pcs
        o3d.visualization.draw_geometries(pcs)
                
    def visualize(self, return_pc=False, vizualize_surfaces=False, surface_id=None, 
                 obb_color=[0, 0.5, 0]):
        """
        Visualize the object with its bounding box and normals.
        
        Creates visualization showing:
        - Point cloud (gray)
        - Oriented bounding box (green by default)
        - Corner spheres
        - Normal vectors (red arrows for sides, green arrow for front)
        - Surface contours and normals (orange) if vizualize_surfaces=True
        
        Args:
            return_pc: If True, return geometries instead of displaying
            vizualize_surfaces: Whether to show surface normals
            surface_id: Specific surface to visualize (None = all)
            obb_color: Color for bounding box [R, G, B]
            
        Returns:
            List of Open3D geometries if return_pc=True
        """
        # Create point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points)
        point_cloud.colors = o3d.utility.Vector3dVector(np.ones_like(self.points) * np.array(obb_color))

        # Create oriented bounding box
        oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(self.corners)
        )
        oriented_bbox.color = obb_color
        vis_objects = [oriented_bbox]
        
        # Add corner spheres
        for corner in self.corners_og:
            sphere = create_sphere(color=obb_color, radius=0.02)
            sphere.translate(corner)
            vis_objects.append(sphere)
        
        pcs = []

        # Visualize normals
        for face, normal in zip(self.faces, self.normals):
            face_center = np.mean(self.corners[face], axis=0)
            
            if np.any(self.dominant_normal != normal):
                # Regular normal (red arrow)
                lines = []
                colors = []
                normals_origin = [face_center]
                normal_endpoint = face_center + normal * 0.1
                colors.append([1, 0, 0])
                lines.append([0, 1])
                normals_origin.append(normal_endpoint)
                
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(normals_origin),
                    lines=o3d.utility.Vector2iVector(lines)
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                pcs.append(line_set)
            else:
                # Dominant normal (green arrow, larger)
                arrow_radius = 0.08
                arrow_height = 0.2
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cone_radius=arrow_radius,
                    cone_height=arrow_height,
                    cylinder_radius=arrow_radius / 2,
                    cylinder_height=arrow_height * 2
                )
                
                # Rotate arrow to align with normal
                z_axis = np.array([0, 0, 1])
                axis = np.cross(z_axis, normal)
                if np.linalg.norm(axis) < 1e-6:
                    rotation_matrix = np.eye(3)
                else:
                    axis /= np.linalg.norm(axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    
                arrow.rotate(rotation_matrix, center=np.array((0, 0, 0)))
                arrow.translate(face_center)
                arrow.paint_uniform_color(obb_color)
                arrow.compute_vertex_normals()
                pcs.append(arrow)

        # Add opposite points visualization
        o_pc = o3d.geometry.PointCloud()
        o_pc.points = o3d.utility.Vector3dVector(self.dominant_normal_opposite_points.numpy())
        o_pc.colors = o3d.utility.Vector3dVector(
            np.array([0.0, 0.3, 0.0]) * np.ones_like(self.dominant_normal_opposite_points.numpy())
        )
        pcs.append(o_pc)
        
        pcs.extend([point_cloud] + vis_objects)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        pcs.append(coordinate_frame)
        
        # Visualize surfaces if requested
        if vizualize_surfaces:
            arrow_radius = 0.08
            arrow_height = 0.2
            surfaces = [self.surfaces[surface_id]] if surface_id is not None else self.surfaces
            
            for s in surfaces:
                s_normals = s['contour normals'].numpy()
                s_c_points = s['contour points']
                s_c_points = s_c_points.numpy() if not isinstance(s_c_points, np.ndarray) else s_c_points
                
                # Draw normal arrow for each contour point
                for p, n in zip(s_c_points, s_normals):
                    arrow = o3d.geometry.TriangleMesh.create_arrow(
                        cone_radius=arrow_radius / 4,
                        cone_height=arrow_height / 4,
                        cylinder_radius=arrow_radius / 8,
                        cylinder_height=arrow_height / 2
                    )
                    z_axis = np.array([0, 0, 1])
                    normal = n
                    
                    # Rotate arrow
                    axis = np.cross(z_axis, normal)
                    if np.linalg.norm(axis) < 1e-6:
                        rotation_matrix = np.eye(3)
                    else:
                        axis /= np.linalg.norm(axis)
                        angle = np.arccos(np.clip(np.dot(z_axis, normal), -1.0, 1.0))
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                        
                    arrow.rotate(rotation_matrix, center=np.array((0, 0, 0)))
                    arrow.translate(p)
                    arrow.paint_uniform_color([1.0, 0.647, 0.])  # Orange
                    arrow.compute_vertex_normals()
                    pcs.append(arrow)
                    
        if return_pc:
            return pcs
        o3d.visualization.draw_geometries(pcs)