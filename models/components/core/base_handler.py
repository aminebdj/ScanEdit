import numpy as np
import open3d as o3d
import torch
import os
import pymeshlab
import copy
from models.components.utils.geometry_utils import rotation_matrix_from_vector_to_z
from models.components.utils.io_utils import load_yaml



class BaseSceneHandler:
    """Base class for scene handling with core initialization"""
    
    def __init__(self, path_2_mesh, datatype, class_agnostic_masks, class_names, 
                 scene_name='unknown', dataset='replica', path_to_annotations=''):
        self.scene_name = scene_name
        self.datatype = datatype
        self.path_2_mesh = path_2_mesh
        self.class_agnostic_masks = class_agnostic_masks
        self.class_names = class_names
        self.dataset = dataset
        
        # Load annotations
        self.instance_level_annotations = load_yaml(path_to_annotations) if os.path.exists(path_to_annotations) else {}
        
        # Initialize data structures
        self.corrupt_masks = []
        self.corrupt_wall_ids = []
        self.wall_ids = []
        self.error_while_initilizing_objects = []
        
        # Load and process mesh data
        self._load_mesh_data()
        self._align_to_z_axis()
        self._process_class_names()
        self._identify_corrupt_masks()
        
    def _load_mesh_data(self):
        """Load mesh data from file"""
        data = pymeshlab.MeshSet()
        data.load_new_mesh(self.path_2_mesh)
        
        self.face_matrix = data.current_mesh().face_matrix()
        self.normal_matrix = data.current_mesh().vertex_normal_matrix()
        self.vertex_matrix = data.current_mesh().vertex_matrix()
        
        # Handle vertex colors
        if data.current_mesh().vertex_color_matrix().shape[1] == 4:
            self.color_matrix = data.current_mesh().vertex_color_matrix()[:, :3]
        else:
            self.color_matrix = np.ones_like(self.vertex_matrix)
            
    def _align_to_z_axis(self):
        """Align scene to Z-axis based on floor normal"""
        self.floor_id = self.class_names.index('floor')
        floor_normal = self.normal_matrix[self.class_agnostic_masks[self.floor_id]].mean(axis=0)
        self.z_up_align_rotation_matrix = rotation_matrix_from_vector_to_z(floor_normal)
        
        # Apply rotation
        self.vertex_matrix = (self.z_up_align_rotation_matrix @ self.vertex_matrix.T).T
        self.normal_matrix = (self.z_up_align_rotation_matrix @ self.normal_matrix.T).T
        
    def _process_class_names(self):
        """Process and reorder class names for replica dataset"""
        if self.dataset == 'replica':
            class_names_ordered = []
            masks_ordered = []
            for i, c in enumerate(self.class_names):
                if c in ['floor', 'wall']:
                    class_names_ordered.insert(0, c)
                    masks_ordered.insert(0, self.class_agnostic_masks[i])
                else:
                    class_names_ordered.append(c)
                    masks_ordered.append(self.class_agnostic_masks[i])
            self.class_agnostic_masks = np.stack(masks_ordered)
            self.class_names = class_names_ordered
            self.floor_id = self.class_names.index('floor')
            
    def _identify_corrupt_masks(self):
        """Identify corrupt/invalid masks"""
        # Identify corrupt walls
        all_wall_ids = np.array([id for id, c_n in enumerate(self.class_names) if c_n == 'wall'])
        scene_var = np.var(self.vertex_matrix[:, 2])
        
        for wall_id in all_wall_ids:
            vertices = self.vertex_matrix[self.class_agnostic_masks[wall_id]]
            z_var = np.var(vertices[:, 2])
            if z_var < scene_var / 6:
                self.corrupt_wall_ids.append(wall_id)
            else:
                self.wall_ids.append(wall_id)
                
        self.corrupt_masks += self.corrupt_wall_ids
        
        # Identify masks that are too small
        for m_id, mask in enumerate(self.class_agnostic_masks):
            if mask.sum() < 20:
                self.corrupt_masks.append(m_id)
                
        # Identify masks that are too large relative to scene
        scene_dims = self.vertex_matrix.max(axis=0) - self.vertex_matrix.min(axis=0)
        for m_id, mask in enumerate(self.class_agnostic_masks):
            if mask.sum() < 10:
                self.corrupt_masks.append(m_id)
                continue
            obj_dims = self.vertex_matrix[mask].max(axis=0) - self.vertex_matrix[mask].min(axis=0)
            if mask.sum() < 5000 and (obj_dims / scene_dims).max() > 0.8:
                self.corrupt_masks.append(m_id)
                
    def create_open3d_mesh(self):
        """Create Open3D mesh from processed data"""
        self.discard_point = self.vertex_matrix.mean(axis=0)
        self.data = o3d.geometry.TriangleMesh()
        self.data.vertices = o3d.utility.Vector3dVector(self.vertex_matrix)
        self.data.triangles = o3d.utility.Vector3iVector(self.face_matrix)
        self.data.vertex_colors = o3d.utility.Vector3dVector(self.color_matrix)
        self.data.vertex_normals = o3d.utility.Vector3dVector(self.normal_matrix)
        self.data_full = copy.copy(self.data)
        
    def initialize_scene_properties(self):
        """Initialize scene properties like floor, walls, normals"""
        vertices = self.vertex_matrix
        normals = self.normal_matrix
        
        # Floor properties
        self.floor_z = vertices[self.class_agnostic_masks[self.floor_id]][:, -1].mean(0)
        self.floor_normal = normals[self.class_agnostic_masks[self.floor_id]].mean(axis=0)
        self.floor_normal /= np.linalg.norm(self.floor_normal)
        
        # Wall properties
        if self.wall_ids:
            wall_pc_count = [vertices[self.class_agnostic_masks[w_i]].shape[0] for w_i in self.wall_ids]
            sorted_wall_counts = sorted(wall_pc_count, reverse=True)
            second_largest_count = sorted_wall_counts[1] if len(sorted_wall_counts) > 1 else sorted_wall_counts[0]
            dominant_wall_id = self.wall_ids[wall_pc_count.index(second_largest_count)]
            
            self.wall_normal = normals[self.class_agnostic_masks[dominant_wall_id]].mean(axis=0)
            self.wall_normal /= np.linalg.norm(self.wall_normal)
        
        # Scene center
        self.scene_center = np.mean(vertices, axis=0)
        
    def filter_walls(self):
        """Filter walls based on normal similarity"""
        norms = np.linalg.norm(self.data.vertex_normals, axis=-1)
        normalized_normals = (np.asarray(self.data.vertex_normals)[norms != 0]) / (norms[norms != 0])[:, None]
        walls_normals = np.stack([normalized_normals[mask[norms != 0]].mean(axis=0) 
                                 for mask in self.class_agnostic_masks[self.wall_ids]])
        walls_normals_sim = walls_normals[:, :2] @ (walls_normals[:, :2].T)
        walls_normals_sim = walls_normals_sim > 0.5
        
        keep = []
        for sim_i in walls_normals_sim:
            similar_walls = np.where(sim_i)[0]
            walls_normals_z = np.abs(walls_normals[similar_walls][:, -1])
            if len(walls_normals_z) == 0:
                continue
            best_wall_id = np.argmin(walls_normals_z)
            best_wall_id_global = self.wall_ids[similar_walls.tolist()[best_wall_id]]
            if best_wall_id_global not in keep:
                keep.append(best_wall_id_global)
        self.wall_ids = keep
        
    def clean_mesh(self):
        """Remove unmasked vertices from mesh"""
        mask = self.class_agnostic_masks.sum(axis=0) == 0
        vertices = np.asarray(self.data.vertices)
        vertices[mask] = self.discard_point
        
        point_indx = np.arange(len(vertices))
        inst_indices = point_indx[mask]
        faces = np.asarray(self.data.triangles)
        incl_mask = np.isin(faces, inst_indices).sum(axis=-1)
        incl_mask = incl_mask == 0
        faces = faces[incl_mask]
        
        self.data.vertices = o3d.utility.Vector3dVector(vertices)
        self.data.triangles = o3d.utility.Vector3iVector(faces)
        
    def get_scene_boundries(self):
        """Get min and max boundaries of the scene"""
        pc = np.asarray(self.data.vertices)
        return [np.min(pc, axis=0).tolist(), np.max(pc, axis=0).tolist()]
    
    def reset_data(self):
        """Reset data to original copy"""
        self.data = None
        self.data = copy.deepcopy(self.data_copy)
        
    def save_point_cloud(self, save_path="./output/out.ply"):
        """Save point cloud to file"""
        if self.datatype == 'mesh':
            o3d.io.write_triangle_mesh(save_path, self.data)
        elif self.datatype == 'point cloud':
            o3d.io.write_point_cloud(save_path, self.data)
        else:
            print("[ERROR] output is not saved, please check input 3D scene folder.")