import numpy as np
import os
import cv2
import copy
import torch
import open3d as o3d
from pathlib import Path
from models.components.core.base_handler import BaseSceneHandler
from models.components.core.object_manager import ObjectManager
from models.components.rendering.renderer import SceneRenderer
from models.components.rendering.visualization import Visualizer
from models.components.optimization.optimizer import SceneOptimizer
from models.components.utils.scene_graph import SceneGraphBuilder
from models.components.utils.io_utils import IOUtils
from models.components.geometry.mesh_operations import MeshOperations, DummyDataManager
from models.components.geometry.surface_utils import SurfaceUtils

class SCENE_HANDLER(BaseSceneHandler):
    """
    Main point cloud handler that composes all components
    """
    
    def __init__(self, path_2_mesh, datatype, class_agnostic_masks, class_names,
                 scene_name='unknown', text_prompt='unknown', vlm=None, dataset='replica',
                 path_to_annotations='', save_video=False, use_preds=False,
                 number_of_samples=5, folder_name='ours_llm', output_dir = './outputs'):
        
        # Initialize base class
        super().__init__(
            path_2_mesh, datatype, class_agnostic_masks, class_names,
            scene_name, dataset, path_to_annotations
        )
        
        # Additional attributes
        self.vlm = vlm
        self.number_of_samples = number_of_samples
        self.use_preds = use_preds
        self.previous_chat = ""
        self.save_video = save_video
        self.text_prompt = text_prompt
        self.trainable_groups = []
        self.trainable_objects = []
        self.maintain_distance = {}
        self.trainable_object_to_support = {}
        
        # Setup paths
        self.save_in = output_dir
        # self.save_in_out_steps = os.path.join(
        #     output_dir, 'opt_steps'
        # )
        os.makedirs(self.save_in, exist_ok=True)
        # os.makedirs(self.save_in_out_steps, exist_ok=True)
        
        # Initialize mesh data
        self.floor_mesh = None
        self.create_open3d_mesh()
        self.initialize_scene_properties()
        self.filter_walls()
        self.clean_mesh()
        
        # Initialize components
        self.object_manager = ObjectManager(self)
        self.renderer = SceneRenderer(self)
        self.visualizer = Visualizer(self)
        self.optimizer = SceneOptimizer(self)
        self.scene_graph_builder = SceneGraphBuilder(self)
        self.dummy_data_manager = DummyDataManager(self)
        
        # Infer additional masks if needed
        # if dataset != 'replica':
        #     self.infer_aditional_masks(use_preds=use_preds)
            
        # Setup room rendering
        self.room_rendering_mask = np.zeros(len(np.asarray(self.data.vertices))) == 1
        self.scene_view, self.view_min_pt, self.view_max_pt, \
        self.global_view_width, self.global_view_height, self.cameras = self.render_room()
        
        # Load/create objects
        objects_loaded = self.set_objects()
        
        # Update room rendering mask
        max_min_elevation_wall = min([
            self.object_manager.objects[wall_id].points[:, 2].max() for wall_id in self.wall_ids
        ])
        vertices = np.asarray(self.data.vertices)
        self.room_rendering_mask = vertices[:, -1] > (
            max_min_elevation_wall - 
            (max_min_elevation_wall - vertices[:, -1].min()) * 0.15
        )
        self.scene_view, self.view_min_pt, self.view_max_pt, \
        self.global_view_width, self.global_view_height, self.cameras = self.render_room()
        
        # Build scene graph if objects weren't loaded
        self.objects = self.object_manager.objects
        
        if not objects_loaded:
            self.set_scene_graph()
        
        self.set_floor_wall_mesh()
            
        self.data_copy = copy.deepcopy(self.data)
        
    # Delegate methods to components
    def set_objects(self):
        """Initialize objects from masks"""
        return self.object_manager.set_objects(self.save_in, self.use_preds)
        
    def append_vlm_annotations(self, skip_annotations=False):
        """Append VLM annotations to objects"""
        self.object_manager.append_vlm_annotations(skip_annotations)
        
    def get_linked_groups(self):
        """Get linked object groups"""
        return self.object_manager.get_linked_groups()
        
    def get_group_details(self, group_ids):
        """Get group dimensions"""
        return self.object_manager.get_group_details(group_ids)
        
    def render_object(self, *args, **kwargs):
        """Render specific objects"""
        return self.renderer.render_object(*args, **kwargs)
        
    def render_room(self, *args, **kwargs):
        """Render entire room"""
        return self.renderer.render_room(*args, **kwargs)
        
    def render_depth(self, *args, **kwargs):
        """Render depth map"""
        return self.renderer.render_depth(*args, **kwargs)
        
    def render_frames(self, *args, **kwargs):
        """Render multiple frames"""
        return self.renderer.render_frames(*args, **kwargs)
        
    def draw_bboxes_from_camera(self, *args, **kwargs):
        """Draw bounding boxes"""
        return self.visualizer.draw_bboxes_from_camera(*args, **kwargs)
        
    def draw_group_bboxes_from_camera(self, *args, **kwargs):
        """Draw group bounding boxes"""
        return self.visualizer.draw_group_bboxes_from_camera(*args, **kwargs)
        
    def optimize_scene(self):
        """Optimize scene"""
        return self.optimizer.optimize_scene()
        
    def set_scene_graph(self):
        """Build scene graph"""
        self.scene_graph_builder.set_scene_graph()
        
    def get_scene_graph(self):
        """Get scene graph"""
        return self.scene_graph_builder.get_scene_graph()
        
    def initlize_dummy_data(self, num_dummies):
        """Initialize dummy data"""
        self.dummy_data_manager.initialize_dummy_data(num_dummies)
        
    def remove_dummy_data(self):
        """Remove dummy data"""
        self.dummy_data_manager.remove_dummy_data()
        
    def load_functionalies(self):
        """Load functionalities"""
        IOUtils.load_functionalities(self)
        
    def update_functionalities(self):
        """Update functionalities"""
        IOUtils.update_functionalities(self)
        
    def log_pose(self, *args, **kwargs):
        """Log poses"""
        IOUtils.log_pose(self, *args, **kwargs)
        
    def update_point_cloud(self, m_id, T, m_center, create_new=False, update_dummy=None):
        """Update point cloud with transformation"""
        if update_dummy is None:
            self.objects[m_id].update_object_info(
                np.asarray(self.data.vertices)[self.class_agnostic_masks[m_center]].mean(axis=0),
                T
            )
            self.scene_graph_builder._set_objects_image_loc(self.objects[m_id])
            
        data = self.data_dummy[update_dummy] if update_dummy is not None else self.data
        MeshOperations.update_vertices(
            data, self.class_agnostic_masks[m_id], T,
            self.class_agnostic_masks[m_center]
        )
        MeshOperations.update_normals(data, self.class_agnostic_masks[m_id], T)
        MeshOperations.update_faces(data, self.class_agnostic_masks[m_id])
    def save_scene(self, filepath):
        """
        Save the current scene to a PLY file.
        
        Args:
            filepath (str): Path where the PLY file should be saved
        """
        
        
        # Ensure the output directory exists
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the Open3D mesh directly
        if hasattr(self, 'data') and self.data is not None:
            o3d.io.write_triangle_mesh(filepath, self.data)
            print(f"Scene saved to: {filepath}")
        else:
            print(f"Warning: No mesh data found to save")
            # Create and save an empty mesh if no data exists
            empty_mesh = o3d.geometry.TriangleMesh()
            o3d.io.write_triangle_mesh(filepath, empty_mesh)

    def remove_objects(self):
        """Remove objects from mesh"""
        MeshOperations.remove_objects(self.data, self.remove, self.scene_center)
        
    # Additional utility methods
    def infer_aditional_masks(self, use_preds=False):
        """Infer additional masks using connected components"""
        from models.components.geometry.mesh_operations import get_complementary_mesh, oversegment_cc_and_plot
        
        file_name = 'masks_classes.pt' if not use_preds else 'masks_classes_mask3d_with_cc.pt'
        if os.path.exists(os.path.join(self.save_in, file_name)):
            self.class_names, self.class_agnostic_masks = torch.load(
                os.path.join(self.save_in, file_name)
            )
        else:
            compl_scene_mask = self.class_agnostic_masks.sum(0) == 0
            compl_mesh = get_complementary_mesh(self.data, compl_scene_mask)
            new_masks = oversegment_cc_and_plot(compl_mesh)
            self.class_agnostic_masks = np.concatenate(
                [self.class_agnostic_masks, new_masks], axis=0
            )
            self.class_names.extend([f"cc_{i}" for i in range(len(new_masks))])
            torch.save(
                (self.class_names, self.class_agnostic_masks),
                os.path.join(self.save_in, file_name)
            )
            
    def set_floor_wall_mesh(self):
        """Set floor and wall meshes"""
        os.makedirs(
            f'/home/boudjoghra/projects/pc_pred/output/scannetpp/{self.scene_name}/{self.text_prompt}',
            exist_ok=True
        )
        
        classes = []
        class_ids = [idx for idx, cls_name in enumerate(self.class_names) 
                    if cls_name in classes]
        
        for obj_id in class_ids:
            for s_id in range(len(self.objects[obj_id].surfaces)):
                mesh, plane_segment = self.set_surface_mesh(
                    surface_name=self.class_names[obj_id],
                    obj_id=obj_id, surface_id=s_id, return_mesh=True
                )
                self._assign_color(obj_id, mesh, s_id, plane_segment)
                
        if os.path.exists(os.path.join(self.save_in, 'floor.ply')):
            import open3d as o3d
            self.floor_mesh = o3d.io.read_triangle_mesh(
                os.path.join(self.save_in, 'floor.ply')
            )
        else:
            self.set_surface_mesh(surface_name='floor', obj_id=self.floor_id)
            
    def set_surface_mesh(self, surface_name, obj_id, surface_id=0,
                        extension='floor.ply', return_mesh=False):
        """Set surface mesh for floor or walls"""
        import open3d as o3d
        import torch
        
        plane_segment = None
        
        if surface_name != 'wall':
            R = self.objects[obj_id].surfaces[surface_id]['R']
            contour_points = self.objects[obj_id].surfaces[surface_id]['contour points'].numpy()
            contour_points = contour_points @ R.T
            contour_normals = self.objects[obj_id].surfaces[surface_id]['contour normals'].numpy()
        else:
            # Handle wall surface
            mask = self.class_agnostic_masks[obj_id]
            surface_points = np.array(self.data.vertices)[mask]
            inlier_mask, R, surface_normal = SurfaceUtils.segment_largest_plane_and_align(
                surface_points, scene_center=self.scene_center
            )
            plane_segment = inlier_mask
            
            # Extract contour
            contour_points, contour_normals = SurfaceUtils.extract_wall_contour_from_grid(
                np.array(self.data.vertices) @ R.T, mask
            )
            
            floor_height = np.array(self.data.vertices)[mask][inlier_mask][:, 2].mean()
            contour_points = np.concatenate([
                contour_points,
                np.ones((contour_points.shape[0], 1)) * floor_height
            ], axis=-1)
            contour_normals = np.concatenate([
                -contour_normals,
                np.zeros((contour_normals.shape[0], 1))
            ], axis=-1)
            
        # Create mesh from contour
        mesh = SurfaceUtils.create_surface_mesh_from_contour(
            contour_points, contour_normals,
            contour_points[:, -1].mean(), R
        )
        
        if return_mesh:
            return mesh, plane_segment
            
        # Save mesh
        output_file_path = os.path.join(self.save_in, extension)
        o3d.io.write_triangle_mesh(output_file_path, mesh)
        
    def _assign_color(self, obj_idx, surface_mesh, surface_idx, plane_segment=None):
        """Assign colors to surface mesh from scene colors"""
        import open3d as o3d
        from tqdm import tqdm
        
        completed_parts = np.asarray(surface_mesh.vertices)
        completed_part_normal = (np.array([0, 0, 1]) 
                                if self.class_names[obj_idx] != 'wall'
                                else np.asarray(surface_mesh.vertex_normals).mean(axis=0))
        
        object_normals = np.asarray(self.data.vertex_normals)[
            self.class_agnostic_masks[obj_idx]
        ]
        mask = (object_normals @ completed_part_normal > 0.6 
               if plane_segment is None else plane_segment)
        scene_vertices = np.asarray(self.data.vertices)[
            self.class_agnostic_masks[obj_idx]
        ][mask]
        scene_colors = np.asarray(self.data.vertex_colors)[
            self.class_agnostic_masks[obj_idx]
        ][mask]
        
        # Assign colors
        color = []
        new_vertices = completed_parts
        for v in tqdm(new_vertices):
            distance = np.abs(scene_vertices - v).sum(axis=-1)
            closest_id = np.argmin(distance)
            
            if distance[closest_id] < 0.05:
                color.append(scene_colors[closest_id])
            else:
                mirror_point = (scene_vertices - v)[closest_id] + scene_vertices[closest_id]
                distance_to_mirror_point = np.abs(scene_vertices - mirror_point).sum(axis=-1)
                closest_id_to_mirror_pt = np.argmin(distance_to_mirror_point)
                color.append(scene_colors[closest_id_to_mirror_pt])
                
        surface_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(color))
        o3d.io.write_triangle_mesh(
            f'/home/boudjoghra/projects/pc_pred/output/scannetpp/{self.scene_name}/'
            f'{self.text_prompt}/new_{self.class_names[obj_idx]}_{obj_idx}_{surface_idx}_mesh.ply',
            surface_mesh
        )
        
    def get_closest_floor_point(self, pixel_query):
        """Get closest floor point from pixel query"""
        pixel_query = np.array(pixel_query)
        clossest_in_quantized = np.argmin(
            np.linalg.norm(pixel_query[None, :] - self.proj_2d, axis=-1)
        )
        return self.floor_points[clossest_in_quantized]
        
    def set_room_grid_points(self):
        """Set grid points for room regions"""
        max_2d_pt = self.proj_2d.max(axis=0)
        min_2d_pt = self.proj_2d.min(axis=0)
        
        # Create 3x3 grid divisions
        x_divisions = np.linspace(min_2d_pt[0], max_2d_pt[0], 4)
        y_divisions = np.linspace(min_2d_pt[1], max_2d_pt[1], 4)
        
        region_map = {
            'top left': (0, 0), 'top': (0, 1), 'top right': (0, 2),
            'left': (1, 0), 'center': (1, 1), 'right': (1, 2),
            'bottom left': (2, 0), 'bottom': (2, 1), 'bottom right': (2, 2)
        }
        
        furthest_indices = {}
        region_center = (max_2d_pt + min_2d_pt) / 2
        
        for region_name, (j, i) in region_map.items():
            # Grid region boundaries
            x_min, x_max = x_divisions[i], x_divisions[i + 1]
            y_min, y_max = y_divisions[j], y_divisions[j + 1]
            
            # Find points in current grid region
            mask = (
                (self.proj_2d[:, 0] >= x_min) & (self.proj_2d[:, 0] < x_max) &
                (self.proj_2d[:, 1] >= y_min) & (self.proj_2d[:, 1] < y_max)
            )
            points_in_region = self.proj_2d[mask]
            
            if len(points_in_region) == 0:
                furthest_indices[region_name] = None
                continue
                
            # Calculate distances to region center
            distances = np.linalg.norm(points_in_region - region_center, axis=1)
            
            # Find furthest point (or nearest for center)
            if region_name == 'center':
                furthest_point_idx = np.argmin(distances)
            else:
                furthest_point_idx = np.argmax(distances)
                
            furthest_indices[region_name] = self.floor_points[
                np.where(mask)[0][furthest_point_idx]
            ]
            
        self.region_to_3d = furthest_indices
        
    def get_image_camera_for_object_or_region(self, object_or_region, region_type, 
                                              group, use_dummy_data=True):
        """
        Get image and camera for object or region
        
        Args:
            object_or_region: Region name or object ID
            region_type: Type of region ('top', 'inside', etc.)
            group: Group of objects
            use_dummy_data: Whether to use dummy data
            
        Returns:
            Dictionary with image, camera, surface, and visibility info
        """
        linked_groups = self.get_linked_groups()
        keep_other_objects = len(group + linked_groups[group[0]]) == 1
        
        if object_or_region in self.region_to_3d.keys():
            # Region-based rendering
            surface = self.floor_points
            surface_ids = np.zeros(len(surface)).astype(int)
            out = self.render_room(use_dummy_data=use_dummy_data)
            object_in_global_with_bbox = self.draw_group_bboxes_from_camera(
                group + linked_groups[group[0]], (out[-1], out[1][:2]), 
                out[0], [0, 0.6, 0.0]
            )
            
            if object_or_region != 'center':
                look_from = (self.region_to_3d['center'] + 
                           np.array([0, 0, self.scene_center[-1] - self.floor_z]) * 1.8)
                dir_vec = self.region_to_3d[object_or_region] - look_from
                look_from += 0.5 * dir_vec
                out = self.render_object(
                    [0], keep_other_objects=True,
                    look_at_point=self.region_to_3d[object_or_region],
                    look_from=look_from, use_dummy_data=use_dummy_data
                )
                depth = self.render_object(
                    [0], keep_other_objects=True,
                    look_at_point=self.region_to_3d[object_or_region],
                    look_from=look_from, use_dummy_data=use_dummy_data, get_depth=True
                )
            else:
                out = self.render_room(use_dummy_data=use_dummy_data, region='center')
                depth = self.render_room(use_dummy_data=use_dummy_data, region='center', get_depth=True)
                
            out_front_obj = self.render_object(
                render_only=group + linked_groups[group[0]],
                keep_other_objects=keep_other_objects,
                use_dummy_data=use_dummy_data, pad=True, pad_percent=1, add_half_height=True
            )
            image_without_box = out_front_obj[0]
            image_with_box = self.draw_group_bboxes_from_camera(
                group + linked_groups[group[0]], (out_front_obj[-1], out_front_obj[1][:2]),
                out_front_obj[0], [0, 0.6, 0.0]
            )
            
            projected_points_shifted, mask_in_view = return_visible_projected(
                depth, out, surface
            )
            
            return {
                'image_camera': out, 'surface': surface,
                'mask_in_view_visible': mask_in_view,
                'projected_points_shifted': projected_points_shifted,
                'surface_ids': surface_ids, 'object_image': image_without_box,
                'object_image_with_box': image_with_box,
                'in_global_with_bbox': object_in_global_with_bbox
            }
            
        elif object_or_region in range(len(self.class_names)):
            # Object-based rendering
            if region_type == 'top':
                out = self.render_room(use_dummy_data=use_dummy_data)
                object_in_global_with_bbox = self.draw_group_bboxes_from_camera(
                    group + linked_groups[group[0]], (out[-1], out[1][:2]),
                    out[0], [0, 0.6, 0.0]
                )
                
                surface = self.objects[object_or_region].surfaces[0]['free points'].float().numpy()
                surface_ids = np.zeros(len(surface)).astype(int)
                
                out = self.render_object(
                    render_only=[object_or_region], render_front=False, pad=False,
                    keep_other_objects=True, use_dummy_data=use_dummy_data
                )
                depth = self.render_object(
                    render_only=[object_or_region], render_front=False, pad=False,
                    keep_other_objects=True, use_dummy_data=use_dummy_data, get_depth=True
                )
                
                out_front_obj = self.render_object(
                    render_only=group + linked_groups[group[0]],
                    keep_other_objects=keep_other_objects,
                    use_dummy_data=use_dummy_data, pad=True, pad_percent=1, add_half_height=True
                )
                image_without_box = out_front_obj[0]
                image_with_box = self.draw_group_bboxes_from_camera(
                    group + linked_groups[group[0]], (out_front_obj[-1], out_front_obj[1][:2]),
                    out_front_obj[0], [0, 0.6, 0.0]
                )
                
                projected_points_shifted, mask_in_view = return_visible_projected(
                    depth, out, surface
                )
                
                return {
                    'image_camera': out, 'surface': surface,
                    'mask_in_view_visible': mask_in_view,
                    'projected_points_shifted': projected_points_shifted,
                    'surface_ids': surface_ids, 'object_image': image_without_box,
                    'object_image_with_box': image_with_box,
                    'in_global_with_bbox': object_in_global_with_bbox
                }
                
            elif region_type == 'inside':
                out = self.render_room(use_dummy_data=use_dummy_data)
                object_in_global_with_bbox = self.draw_group_bboxes_from_camera(
                    group + linked_groups[group[0]], (out[-1], out[1][:2]),
                    out[0], [0, 0.6, 0.0]
                )
                
                surface = np.concatenate([
                    s['free points'].float().numpy()
                    for s in self.objects[object_or_region].surfaces
                ])
                surface_ids = np.concatenate([
                    (np.ones(len(s['free points'].float().numpy())) * s_id).astype(int)
                    for s_id, s in enumerate(self.objects[object_or_region].surfaces)
                ])
                
                out = self.render_object(
                    render_only=[object_or_region], pad=False,
                    keep_other_objects=True, use_dummy_data=use_dummy_data
                )
                depth = self.render_object(
                    render_only=[object_or_region], pad=False,
                    keep_other_objects=True, use_dummy_data=use_dummy_data, get_depth=True
                )
                
                out_front_obj = self.render_object(
                    render_only=group + linked_groups[group[0]],
                    keep_other_objects=keep_other_objects,
                    use_dummy_data=use_dummy_data, pad=True, pad_percent=1, add_half_height=True
                )
                image_without_box = out_front_obj[0]
                image_with_box = self.draw_group_bboxes_from_camera(
                    group + linked_groups[group[0]], (out_front_obj[-1], out_front_obj[1][:2]),
                    out_front_obj[0], [0, 0.6, 0.0]
                )
                
                projected_points_shifted, mask_in_view = return_visible_projected(
                    depth, out, surface
                )
                
                return {
                    'image_camera': out, 'surface': surface,
                    'mask_in_view_visible': mask_in_view,
                    'projected_points_shifted': projected_points_shifted,
                    'surface_ids': surface_ids, 'object_image': image_without_box,
                    'object_image_with_box': image_with_box,
                    'in_global_with_bbox': object_in_global_with_bbox
                }
            else:
                surface = self.floor_points
                surface_ids = np.zeros(len(surface)).astype(int)
                
                out = self.render_room(use_dummy_data=use_dummy_data)
                object_in_global_with_bbox = self.draw_group_bboxes_from_camera(
                    group + linked_groups[group[0]], (out[-1], out[1][:2]),
                    out[0], [0, 0.6, 0.0]
                )
                
                out = self.render_object(
                    render_only=[object_or_region], render_front=False, pad=True,
                    keep_other_objects=True, use_dummy_data=use_dummy_data, pad_percent=0.25
                )
                
                out_front_obj = self.render_object(
                    render_only=group + linked_groups[group[0]],
                    keep_other_objects=keep_other_objects,
                    use_dummy_data=use_dummy_data, pad=True, pad_percent=1, add_half_height=True
                )
                image_without_box = out_front_obj[0]
                image_with_box = self.draw_group_bboxes_from_camera(
                    group + linked_groups[group[0]], (out_front_obj[-1], out_front_obj[1][:2]),
                    out_front_obj[0], [0, 0.6, 0.0]
                )
                
                depth = self.render_object(
                    render_only=[object_or_region], render_front=False, pad=True,
                    keep_other_objects=True, use_dummy_data=use_dummy_data,
                    pad_percent=0.5, get_depth=True
                )
                
                projected_points_shifted, mask_in_view = return_visible_projected(
                    depth, out, surface
                )
                
                return {
                    'image_camera': out, 'surface': surface,
                    'mask_in_view_visible': mask_in_view,
                    'projected_points_shifted': projected_points_shifted,
                    'surface_ids': surface_ids, 'object_image': image_without_box,
                    'object_image_with_box': image_with_box,
                    'in_global_with_bbox': object_in_global_with_bbox
                }
        else:
            # Default case
            surface = self.floor_points
            surface_ids = np.zeros(len(surface)).astype(int)
            
            out = self.render_room(use_dummy_data=use_dummy_data)
            object_in_global_with_bbox = self.draw_group_bboxes_from_camera(
                group + linked_groups[group[0]], (out[-1], out[1][:2]),
                out[0], [0, 0.6, 0.0]
            )
            
            out_front_obj = self.render_object(
                render_only=group + linked_groups[group[0]],
                keep_other_objects=keep_other_objects,
                use_dummy_data=use_dummy_data, pad=True, pad_percent=1, add_half_height=True
            )
            image_without_box = out_front_obj[0]
            image_with_box = self.draw_group_bboxes_from_camera(
                group + linked_groups[group[0]], (out_front_obj[-1], out_front_obj[1][:2]),
                out_front_obj[0], [0, 0.6, 0.0]
            )
            
            depth = self.render_room(use_dummy_data=use_dummy_data, get_depth=True)
            projected_points_shifted, mask_in_view = return_visible_projected(
                depth, out, surface
            )
            
            return {
                'image_camera': out, 'surface': surface,
                'mask_in_view_visible': mask_in_view,
                'projected_points_shifted': projected_points_shifted,
                'surface_ids': surface_ids, 'object_image': image_without_box,
                'object_image_with_box': image_with_box,
                'in_global_with_bbox': object_in_global_with_bbox
            }
            
    def get_max_hights_for_surfaces(self, obj_id):
        """Get maximum heights for object surfaces"""
        related_objects = {id: [] for id in range(len(self.objects[obj_id].surfaces))} \
                         if self.objects[obj_id].surfaces else {0: []}
        
        if self.objects[obj_id].surfaces:
            surfaces_eles = np.array([
                s['elevation'][-1].item() for s in self.objects[obj_id].surfaces
            ])
            for obj in self.objects:
                if obj.obj_name == 'corrupt':
                    continue
                for c in obj.constraints:
                    if c[0] == 'on top of' and c[-1] == obj_id:
                        related_objects[0].append(obj.get_obj_details(get_position_and_orientation=True))
                    elif c[0] == 'inside' and c[-1] == obj_id:
                        obj_min_point = obj.points[:, 2].min()
                        closest_s_id = np.argmin(np.abs(surfaces_eles - obj_min_point))
                        related_objects[closest_s_id].append(
                            obj.get_obj_details(get_position_and_orientation=True)
                        )
        else:
            for obj in self.objects:
                if obj.obj_name == 'corrupt':
                    continue
                for c in obj.constraints:
                    if c[0] == 'on top of' and c[-1] == obj_id:
                        related_objects[0].append(
                            obj.get_obj_details(get_position_and_orientation=True)
                        )
                        
        # Calculate dimensions
        supported_obj_dims = {}
        supported_groups_dims = {}
        
        for k, v in related_objects.items():
            small, medium, large = [], [], []
            supported_obj_dims[k] = []
            for obj in v:
                dx, dy, dz = obj['dimensions']
                
                if dx < 0.3 and dy < 0.3 and dz < 0.3:
                    small.append((dx, dy, dz))
                elif dx < 1.0 and dy < 1.0 and dz < 1.0:
                    medium.append((dx, dy, dz))
                else:
                    large.append((dx, dy, dz))
                    
                supported_obj_dims[k].append([f"dx = {dx}, dy = {dy}, dz = {dz}"])
                
            supported_groups_dims[k] = (
                f"Number of small objects: {len(small)}, dx < 0.3m, dy < 0.3m, dz < 0.3m | "
                f"Number of medium objects: {len(medium)} | "
                f"Number of large objects: {len(large)}"
            )
            
        surfaces_max_heights = ""
        surfaces = self.objects[obj_id].surfaces
        if surfaces:
            pred_name = (self.objects[obj_id].pred_obj_name[0]
                        if self.objects[obj_id].pred_obj_name
                        else self.objects[obj_id].obj_name)
            if (self.objects[obj_id].obj_name != 'unknown' and
                'cc' not in self.objects[obj_id].obj_name):
                pred_name = self.objects[obj_id].obj_name
                
            surface_cp = (surfaces[0]['contour points'][:, :2]
                         if isinstance(surfaces[0]['contour points'], np.ndarray)
                         else surfaces[0]['contour points'][:, :2].numpy())
            s_dims = np.round(surface_cp.max(axis=0) - surface_cp.min(axis=0), 2)
            
            surfaces_max_heights = (
                f"""The {pred_name} with ID {self.objects[obj_id].obj_id} has """
                f"""{len(self.objects[obj_id].surfaces)} support surfaces, each one can support """
                f"""objects with maximum height z and maximum (x, y) dimensions of: 
            Surface ID 0 (Highest level surface) currently supporting """
                f"""{len(supported_obj_dims[0])} objects with dimensions {supported_groups_dims[0]}: """
                f"""can support any height, maximum dx = {s_dims[0]} meters, maximum dy = {s_dims[1]} meters
            """
            )
            
            if len(surfaces) > 1:
                surfaces_eles = np.array([s['elevation'][-1].item() for s in surfaces])
                surfaces_eles = surfaces_eles[:-1] - surfaces_eles[1:]
                for s_id, max_height in enumerate(surfaces_eles):
                    surface_cp = (surfaces[s_id + 1]['contour points'][:, :2]
                                if isinstance(surfaces[s_id + 1]['contour points'], np.ndarray)
                                else surfaces[s_id + 1]['contour points'][:, :2].numpy())
                    s_dims = np.round(surface_cp.max(axis=0) - surface_cp.min(axis=0), 2)
                    note = "" if s_id != len(surfaces_eles) - 1 else "(Lowest level surface)"
                    surfaces_max_heights += (
                        f"\n Surface ID {s_id + 1} {note}, currently supporting "
                        f"{len(supported_obj_dims[s_id + 1])} objects with dimensions "
                        f"{supported_groups_dims[s_id + 1]}: maximum height dz = {round(max_height, 2)} meters, "
                        f"maximum dx = {s_dims[0]} meters, maximum dy = {s_dims[1]} meters"
                    )
                    
        return surfaces_max_heights
        
    def get_graph_related_objects_in_ref_frame(self, id, exclude_ids=[], 
                                               trasn_z_angle=None, translate=None,
                                               top_k=5, furthest=True):
        """Get related objects in reference frame"""
        id = int(id)
        related_objects = {}
        
        if self.objects[id].surfaces:
            surfaces_eles = np.array([
                s['elevation'][-1].item() for s in self.objects[id].surfaces
            ])
            
            for obj in self.objects:
                if obj.obj_name == 'corrupt':
                    continue
                if obj.obj_id in exclude_ids:
                    continue
                    
                on_top, inside = False, False
                if self.objects[id].obj_name == 'floor' and obj.obj_name == 'door':
                    key = f'on top of the {self.objects[id].obj_name} in surface ID 0'
                    if key not in related_objects.keys():
                        related_objects[key] = []
                    related_objects[key].append(
                        obj.get_obj_details(
                            get_position_and_orientation=True,
                            trasn_z_angle=trasn_z_angle,
                            translate=translate
                        )
                    )
                    on_top = True
                    
                for c in obj.constraints:
                    if c[0] == 'on top of' and c[-1] == id:
                        key = f'on top of the {self.objects[id].obj_name} in surface ID 0'
                        if key not in related_objects.keys():
                            related_objects[key] = []
                        related_objects[key].append(
                            obj.get_obj_details(
                                get_position_and_orientation=True,
                                trasn_z_angle=trasn_z_angle,
                                translate=translate
                            )
                        )
                        on_top = True
                    elif c[0] == 'inside' and c[-1] == id:
                        obj_min_point = obj.points[:, 2].min()
                        closest_s_id = np.argmin(np.abs(surfaces_eles - obj_min_point))
                        key = f'inside of the {self.objects[id].obj_name} in surface ID {closest_s_id}'
                        if key not in related_objects.keys():
                            related_objects[key] = []
                        related_objects[key].append(
                            obj.get_obj_details(
                                get_position_and_orientation=True,
                                trasn_z_angle=trasn_z_angle,
                                translate=translate
                            )
                        )
                        inside = True
                        
                if (obj.obj_id != id and not on_top and not inside and
                    self.objects[id].obj_name != 'floor' and
                    np.linalg.norm(obj.points[::10][None, :, :] -
                                  self.objects[id].points[::10][:, None, :],
                                  axis=-1).min() < 0.3):
                    key = f'close to the {self.objects[id].obj_name}'
                    if key not in related_objects.keys():
                        related_objects[key] = []
                    related_objects[key].append(
                        obj.get_obj_details(
                            get_position_and_orientation=True,
                            trasn_z_angle=trasn_z_angle,
                            translate=translate
                        )
                    )
        else:
            for obj in self.objects:
                if obj.obj_name == 'corrupt':
                    continue
                if obj.obj_id in exclude_ids:
                    continue
                    
                on_top = False
                for c in obj.constraints:
                    if c[0] == 'on top of' and c[-1] == id:
                        key = f'on top of the {self.objects[id].obj_name} in surface ID 0'
                        if key not in related_objects.keys():
                            related_objects[key] = []
                        related_objects[key].append(
                            obj.get_obj_details(
                                get_position_and_orientation=True,
                                trasn_z_angle=trasn_z_angle,
                                translate=translate
                            )
                        )
                        on_top = True
                        
                if (obj.obj_id != id and not on_top and
                    self.objects[id].obj_name != 'floor' and
                    np.linalg.norm(obj.points[::10][None, :, :] -
                                  self.objects[id].points[::10][:, None, :],
                                  axis=-1).min() < 0.3):
                    key = f'close to the {self.objects[id].obj_name}'
                    if key not in related_objects.keys():
                        related_objects[key] = []
                    related_objects[key].append(
                        obj.get_obj_details(
                            get_position_and_orientation=True,
                            trasn_z_angle=trasn_z_angle,
                            translate=translate
                        )
                    )
                    
        return related_objects


# Helper function for visibility checking
def return_visible_projected(depth, out, surface):
    """
    Check which surface points are visible from camera
    
    Args:
        depth: Depth map
        out: Output from render function
        surface: Surface points
        
    Returns:
        Tuple of (projected_points_shifted, mask_in_view)
    """
    import torch
    
    camera = out[-1]
    min_pt = out[1]
    max_pt = np.array(out[0].size) if (np.all(out[2] == 0) or out[2] is None) else out[2]
    
    projected_points = camera.transform_points_screen(
        torch.from_numpy(surface).cuda()
    ).cpu().numpy()
    projected_depth = camera.get_world_to_view_transform().transform_points(
        torch.from_numpy(surface).cuda()
    ).cpu().numpy()[:, 2]
    
    projected_points_shifted = projected_points[:, :2] - min_pt[:2]
    mask_in_view = (
        (projected_points[:, 0] > min_pt[0]) *
        (projected_points[:, 1] > min_pt[1]) *
        (projected_points[:, 0] < max_pt[0]) *
        (projected_points[:, 1] < max_pt[1])
    ) == 1
    
    x_indices = projected_points[mask_in_view, 0]
    y_indices = projected_points[mask_in_view, 1]
    
    # Access depth values
    selected_depth_values = depth[y_indices.astype(int), x_indices.astype(int)]
    visible_points = ((np.abs(projected_depth[mask_in_view] - 0.05) < selected_depth_values))
    mask_in_view[mask_in_view] = visible_points
    
    return projected_points_shifted, mask_in_view