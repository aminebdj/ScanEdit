import numpy as np
import torch
import os
import copy
from tqdm import tqdm
from models.components.geometry.object_point_cloud import OBJ_POINT_CLOUD

from models.components.geometry.transformations import rotation_matrix
from models.components.utils.geometry_utils import rotation_matrix_from_vector_to_z



class ObjectManager:
    """Manages object creation, relationships, and properties"""
    
    def __init__(self, handler):
        self.handler = handler
        self.objects = []
        
    def set_objects(self, save_in='./output/preprocessed', use_preds=False):
        """Initialize objects from masks"""
        import glob
        
        extension = '_preds' if use_preds else ''
        
        # Try to load existing objects
        if os.path.exists(os.path.join(save_in, f'objects{extension}.pt')):
            return self._load_objects_from_file(os.path.join(save_in, f'objects{extension}.pt'))
            
        # Try to load chunked objects
        if os.path.exists(os.path.join(save_in, f'objects{extension}_0.pt')):
            chunk_files = sorted(
                glob.glob(os.path.join(save_in, f"objects{extension}_*.pt")),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            return self._load_objects_from_chunks(chunk_files)
            
        # Create new objects
        return self._create_new_objects(save_in, extension)
        
    def _load_objects_from_file(self, filepath):
        """Load objects from single file"""
        objects_list = torch.load(filepath)
        for obj_dict in objects_list:
            obj_ = OBJ_POINT_CLOUD(
                cameras=self.handler.cameras,
                scene_points=np.array(self.handler.data.vertices),
                empty=True
            )
            obj_.load_from(obj_dict)
            obj_.reset_dominant_normal()
            self.objects.append(obj_)
        self.append_vlm_annotations()
        return True
        
    def _load_objects_from_chunks(self, chunk_files):
        """Load objects from multiple chunk files"""
        objects_list = []
        for file in chunk_files:
            chunk = torch.load(file)
            objects_list.extend(chunk)
            
        for obj_dict in objects_list:
            obj_ = OBJ_POINT_CLOUD(
                cameras=self.handler.cameras,
                scene_points=np.array(self.handler.data.vertices),
                empty=True
            )
            obj_.load_from(obj_dict)
            obj_.reset_dominant_normal()
            self.objects.append(obj_)
        self.append_vlm_annotations()
        return True
        
    def _create_new_objects(self, save_in, extension):
        """Create new objects from scratch"""
        scene_center = np.array(self.handler.data.vertices).mean(0)
        
        for m_id, (mask, class_name) in tqdm(enumerate(zip(
            self.handler.class_agnostic_masks,
            self.handler.class_names
        ))):
            if m_id in self.handler.corrupt_masks:
                self.objects.append(OBJ_POINT_CLOUD(corrupt=True, obj_id=m_id))
                continue
                
            # Try to get annotation
            try:
                annos = self.handler.instance_level_annotations[m_id]
                pred_classes = [c['pred_class_name'] for c in annos.values()]
                confidences = [float(c['confidence']) for c in annos.values()]
                pred_class = max(pred_classes, key=pred_classes.count)
                class_frequency = sum([1 for c in pred_classes if c == pred_class])
                
                if ('cc' in class_name or class_name == 'unknown') and class_frequency != 1:
                    class_name = pred_class
                elif ('cc' in class_name or class_name == 'unknown') and class_frequency == 1:
                    class_name = max(zip(pred_classes, confidences), key=lambda x: x[1])[0]
            except:
                pass
                
            if class_name in ['floor', 'wall']:
                obj = self._create_floor_wall_object(m_id, mask, class_name, scene_center)
            else:
                obj = self._create_regular_object(m_id, mask, class_name, scene_center)
                
            obj.reset_dominant_normal()
            self.objects.append(obj)
            
        self.append_vlm_annotations()
        return False
        
    def _create_floor_wall_object(self, m_id, mask, class_name, scene_center):
        """Create floor or wall object with special handling"""
        surface_points = np.array(self.handler.data.vertices)[mask]
        centroid = np.mean(surface_points, axis=0)
        centered_points = surface_points - centroid
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        surface_normal = eigenvectors[:, np.argmin(eigenvalues)]
        surface_normal = surface_normal * np.sign(
            (-surface_points.mean(0) + scene_center).T @ surface_normal
        )
        
        if class_name == 'wall':
            surface_normal[-1] *= 0
            
        R = rotation_matrix_from_vector_to_z(surface_normal) if class_name == 'wall' else np.eye(3)
        
        th = 1.2 if class_name == 'floor' else 0.2
        vertices = copy.deepcopy(np.array(self.handler.data.vertices)) @ R.T
        floor_height = vertices[mask][:, 2].mean()
        z_vals = vertices[:, 2]
        max_z = vertices[:, 2].max()
        min_z = vertices[:, 2].min()
        points_ref = vertices[z_vals < (floor_height + th * (max_z - min_z))]
        vertices = np.concatenate([
            points_ref[:, :2],
            np.ones((len(points_ref), 1)) * floor_height
        ], axis=-1)
        
        num_points = len(vertices) // 2
        vertices[num_points:, 2] += 0.0001
        vertices[:num_points, 2] -= 0.0001
        
        return OBJ_POINT_CLOUD(
            vertices @ R,
            obj_name=class_name,
            normals=np.ones_like(points_ref) * surface_normal[None, :],
            scene_center=scene_center,
            cameras=self.handler.cameras,
            scene_name=self.handler.scene_name,
            R=R,
            obj_id=m_id
        )
        
    def _create_regular_object(self, m_id, mask, class_name, scene_center):
        """Create regular object"""
        wall_objects = [self.objects[i] for i in self.handler.wall_ids]
        return OBJ_POINT_CLOUD(
            np.array(self.handler.data.vertices)[mask],
            obj_name=class_name,
            normals=np.array(self.handler.data.vertex_normals)[mask],
            scene_center=scene_center,
            cameras=self.handler.cameras,
            scene_name=self.handler.scene_name,
            scene_points=np.array(self.handler.data.vertices),
            obj_id=m_id,
            wall_objects=wall_objects
        )
        
    def append_vlm_annotations(self, skip_annotations=False):
        """Append VLM annotations to objects"""
        for obj in self.objects:
            if obj.obj_name == 'corrupt':
                continue
                
            if obj.obj_id not in self.handler.instance_level_annotations.keys() or skip_annotations:
                obj.confidence = []
                obj.pred_obj_name = []
                obj.colors = []
                obj.material = []
                obj.functionality = []
                obj.description = []
                continue
                
            annotations = self.handler.instance_level_annotations[obj.obj_id]
            
            # Extract and sort by confidence
            obj.confidence = [v['confidence'] for v in annotations.values() if 'confidence' in v.keys()]
            indices = self._get_sorted_indices_desc(obj.confidence)
            obj.confidence = self._sort_with_inds(obj.confidence, indices)
            
            obj.pred_obj_name = [v['pred_class_name'] for v in annotations.values() if 'pred_class_name' in v.keys()]
            obj.pred_obj_name = self._sort_with_inds(obj.pred_obj_name, indices)
            
            obj.colors = [v['color'] for v in annotations.values() if 'color' in v.keys()]
            obj.colors = self._sort_with_inds(obj.colors, indices)
            
            obj.material = [v['material'] for v in annotations.values() if 'material' in v.keys()]
            obj.material = self._sort_with_inds(obj.material, indices)
            
            obj.functionality = [v['functionality'] for v in annotations.values() if 'functionality' in v.keys()]
            obj.functionality = self._sort_with_inds(obj.functionality, indices)
            
            obj.description = [v['description'] for v in annotations.values() if 'description' in v.keys()]
            obj.description = self._sort_with_inds(obj.description, indices)
            
    @staticmethod
    def _get_sorted_indices_desc(lst):
        """Get indices sorted in descending order"""
        return sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
        
    @staticmethod
    def _sort_with_inds(lst, indices):
        """Sort list using given indices"""
        return [lst[i] for i in indices]
        
    def get_linked_groups(self):
        """Get linked object groups based on support relationships"""
        def collect_supported_objects(obj, visited=None):
            if visited is None:
                visited = set()
                
            supported = []
            if obj not in visited and obj.supported_objects:
                visited.add(obj)
                for idx in obj.supported_objects:
                    supported_obj = self.objects[idx]
                    supported.append(supported_obj.obj_id)
                    supported.extend(collect_supported_objects(supported_obj, visited))
            return supported
            
        return [collect_supported_objects(obj) if obj.obj_name != 'corrupt' else [] 
                for obj in self.objects]
                
    def get_group_details(self, group_ids):
        """Get dimensions of a group of objects"""
        group_points = np.concatenate([self.objects[id].points for id in group_ids])
        dims = group_points.max(axis=0) - group_points.min(axis=0)
        return dims.tolist()