import numpy as np
import torch
from tqdm import tqdm
from models.components.geometry.surface_utils import SurfaceUtils
import copy

class SceneGraphBuilder:
    """Builds scene graph showing object relationships"""
    
    def __init__(self, handler):
        self.handler = handler
        
    def set_scene_graph(self):
        """Build scene graph from objects"""
        maximum_point = np.asarray(self.handler.data.vertices)[:, 2].max()
        minimum_point = np.asarray(self.handler.data.vertices)[:, 2].min()
        scene_height = maximum_point - minimum_point
        scene_width = (np.asarray(self.handler.data.vertices)[:, 0].max() -
                      np.asarray(self.handler.data.vertices)[:, 0].min())
        scene_depth = (np.asarray(self.handler.data.vertices)[:, 1].max() -
                      np.asarray(self.handler.data.vertices)[:, 1].min())
        
        # Set object image locations
        print('Setting object image locations...')
        for obj_id, obj in enumerate(self.handler.objects):
            if obj.obj_name == 'corrupt':
                continue
            self._set_objects_image_loc(obj)
            
        walls = [self.handler.objects[i] for i in self.handler.wall_ids]
        
        # Collect remaining objects
        print('Collecting remaining objects...')
        for obj_id, obj in enumerate(self.handler.objects):
            if obj.obj_name == 'corrupt':
                continue
                
            max_scene_point = np.asarray(self.handler.data.vertices)[:, 2].max()
            min_scene_point = np.asarray(self.handler.data.vertices)[:, 2].min()
            min_point = min_scene_point + 0.9 * (max_scene_point - min_scene_point)
            
            if obj.points[:, 2].min() > min_point:
                continue
                
            if (len(obj.constraints) == 0 and 
                obj.obj_name != 'wall' and 
                obj.obj_name != 'floor'):
                self._find_object_support(obj, walls, scene_height)
                
        # Organize floor objects
        floor_objects = self._get_floor_objects()
        sorted_floor_objects = self._sort_floor_objects(floor_objects)
        
        # Collect supported objects
        collected_objects = self._collect_supported_objects()
        collected_objects_merged = self._merge_collected_objects(collected_objects, walls, scene_height)
        
        # Set surfaces
        self._set_object_surfaces(collected_objects_merged)
        
        # Set supported objects
        for i, obj in enumerate(self.handler.objects):
            obj.supported_objects = collected_objects_merged[i]
            
        # Save objects
        self._save_objects()
        
    def _set_objects_image_loc(self, obj):
        """Set object location in image coordinates"""
        projected_center = self.handler.cameras.transform_points_screen(
            obj.points_torch.cuda().float()
        ).cpu().numpy()
        
        start_point = obj.points_torch.cuda().float().mean(dim=0)
        end_point = start_point + obj.dominant_normal_torch.float().cuda()
        projected_dominant_normal = self.handler.cameras.transform_points_screen(
            torch.stack([start_point, end_point])
        ).cpu().numpy()[:, :2] - np.array(self.handler.view_min_pt)[None, :]
        
        pixel_coords = projected_center[:, :2].mean(axis=0).tolist()
        obj.xy_coords = {
            'x': int(pixel_coords[0] - self.handler.view_min_pt[0]),
            'y': int(pixel_coords[1] - self.handler.view_min_pt[1])
        }
        
        # Determine image region
        width_third = self.handler.global_view_width / 3
        height_third = self.handler.global_view_height / 3
        
        point_count_in_region = {
            'top left': 0, 'bottom left': 0, 'left': 0,
            'top right': 0, 'bottom right': 0, 'right': 0,
            'top': 0, 'bottom': 0, 'center': 0
        }
        
        # Count points in each region
        self._count_points_in_regions(
            projected_center, point_count_in_region,
            width_third, height_third
        )
        
        obj.projected_points = projected_center[:, :2] - self.handler.view_min_pt
        obj.projected_dominant_normal = projected_dominant_normal
        
        # Determine image constraints
        region_map = {
            'top left': (0, 0), 'top': (0, 1), 'top right': (0, 2),
            'left': (1, 0), 'center': (1, 1), 'right': (1, 2),
            'bottom left': (2, 0), 'bottom': (2, 1), 'bottom right': (2, 2)
        }
        
        # Find front region
        obj.front_regions = self._find_front_region(
            projected_dominant_normal, region_map, width_third, height_third
        )
        
        # Find main region
        obj.image_constraints = self._find_main_region(
            point_count_in_region, region_map
        )
        
    def _count_points_in_regions(self, projected_center, point_count_in_region,
                                 width_third, height_third):
        """Count points in each image region"""
        min_pt = self.handler.view_min_pt
        
        point_count_in_region['top left'] += (
            (projected_center[:, 0] < min_pt[0] + width_third) *
            (projected_center[:, 1] < min_pt[1] + height_third)
        ).sum()
        
        point_count_in_region['bottom left'] += (
            (projected_center[:, 0] < min_pt[0] + width_third) *
            (projected_center[:, 1] > min_pt[1] + 2 * height_third)
        ).sum()
        
        point_count_in_region['left'] += (
            (projected_center[:, 0] < min_pt[0] + width_third) *
            (projected_center[:, 1] < min_pt[1] + 2 * height_third) *
            (projected_center[:, 1] > min_pt[1] + height_third)
        ).sum()
        
        point_count_in_region['top'] += (
            (projected_center[:, 0] > min_pt[0] + width_third) *
            (projected_center[:, 0] < min_pt[0] + 2 * width_third) *
            (projected_center[:, 1] < min_pt[1] + height_third)
        ).sum()
        
        point_count_in_region['top right'] += (
            (projected_center[:, 0] > min_pt[0] + 2 * width_third) *
            (projected_center[:, 1] < min_pt[1] + height_third)
        ).sum()
        
        point_count_in_region['right'] += (
            (projected_center[:, 0] > min_pt[0] + 2 * width_third) *
            (projected_center[:, 1] > min_pt[1] + height_third) *
            (projected_center[:, 1] < min_pt[1] + 2 * height_third)
        ).sum()
        
        point_count_in_region['bottom right'] += (
            (projected_center[:, 0] > min_pt[0] + 2 * width_third) *
            (projected_center[:, 1] > min_pt[1] + 2 * height_third)
        ).sum()
        
        point_count_in_region['bottom'] += (
            (projected_center[:, 0] > min_pt[0] + width_third) *
            (projected_center[:, 0] < min_pt[0] + 2 * width_third) *
            (projected_center[:, 1] > min_pt[1] + 2 * height_third)
        ).sum()
        
        point_count_in_region['center'] += (
            (projected_center[:, 0] > min_pt[0] + width_third) *
            (projected_center[:, 0] < min_pt[0] + 2 * width_third) *
            (projected_center[:, 1] > min_pt[1] + height_third) *
            (projected_center[:, 1] < min_pt[1] + 2 * height_third)
        ).sum()
        
    def _find_front_region(self, projected_dominant_normal, region_map,
                          width_third, height_third):
        """Find region in front of object"""
        regions_pixels = (
            np.array([v for v in region_map.values()])[:, [1, 0]] *
            np.array([[width_third, height_third]]) +
            np.array([[width_third, height_third]]) / 2
        )
        
        reg_to_center = regions_pixels - projected_dominant_normal[0][None, :]
        distances = np.linalg.norm(reg_to_center, axis=-1)
        reg_to_center /= distances[:, None]
        
        dominant_normal_dir = projected_dominant_normal[1] - projected_dominant_normal[0]
        dominant_normal_dir /= np.linalg.norm(dominant_normal_dir)
        
        sim = reg_to_center @ dominant_normal_dir
        
        if (sim > 0).sum() == 0:
            return None
            
        k = 3
        top_align_indices = np.argsort(sim)[-k:]
        min_dist_index = top_align_indices[np.argmin(distances[top_align_indices])]
        
        return list(region_map.keys())[min_dist_index]
        
    def _find_main_region(self, point_count_in_region, region_map):
        """Find main region where object is located"""
        grid = []
        all_corners = []
        count_sum = 0
        
        for region, count in point_count_in_region.items():
            all_corners.append(np.array(region_map[region]))
            grid.append(np.array(region_map[region]) * count)
            count_sum += count
            
        grid = np.stack(grid)
        all_corners = np.stack(all_corners)
        mean_value = np.sum(grid, axis=0) / count_sum
        region_id = np.argmin(np.linalg.norm(mean_value[None, :] - all_corners, axis=-1))
        
        return list(point_count_in_region.keys())[region_id]
        
    def _find_object_support(self, obj, walls, scene_height):
        """Find what object is supporting the given object"""
        z_points = []
        proximity_box = np.stack([
            obj.points[:, :2].min(axis=0),
            obj.points[:, :2].max(axis=0)
        ])
        center = obj.points[:, :2].mean(axis=0)
        proximity_box = (proximity_box - center) * 10 + center
        
        min_z = obj.points[:, 2].min()
        max_z = obj.points[:, 2].max()
        min_3d_points = obj.points[obj.points[:, 2] < (min_z + 0.1 * (max_z - min_z))]
        min_z_source = obj.points[:, 2].min()
        
        for obj_ in self.handler.objects:
            if obj_.obj_name == 'corrupt':
                continue
            if obj_ == obj:
                z_points.append(float('inf'))
                continue
            if obj_.obj_name == 'wall':
                z_points.append(float('inf'))
                continue
            if (obj.points[:, 2].min() - obj_.points[:, 2].min() < 0.03 and
                obj_.obj_name != 'floor'):
                z_points.append(float('inf'))
                continue
                
            in_proximity_mask = (
                (obj_.points[:, 0] < proximity_box[1, 0]) &
                (obj_.points[:, 1] < proximity_box[1, 1]) &
                (obj_.points[:, 0] > proximity_box[0, 0]) &
                (obj_.points[:, 1] > proximity_box[0, 1]) &
                (obj_.points[:, 2] < min_z_source + 0.05) &
                (obj_.points[:, 2] > min_z_source - 0.2)
            )
            
            if in_proximity_mask.sum() == 0:
                z_points.append(float('inf'))
                continue
                
            if len(obj_.surfaces) != 0:
                s_elevations = np.array([s['elevation'][-1].item() for s in obj_.surfaces])
                if np.abs(obj.points[:, 2].min() - s_elevations).min() > 0.05:
                    z_points.append(float('inf'))
                    continue
                    
            points_in_proximity = obj_.points[in_proximity_mask]
            dir_vec = min_3d_points[:, None, :] - points_in_proximity[None, :, :]
            distances = np.linalg.norm(dir_vec, axis=-1)
            dir_vec /= distances[:, :, None]
            argmin_dists = np.argmin(distances, axis=-1)
            min_distances = distances.min(axis=-1)
            
            min_distances[
                np.take_along_axis(dir_vec[:, :, 2], argmin_dists[:, None], axis=1)[:, 0] < -0.1
            ] = float('inf')
            
            intersection = obj.get_intersection(obj_, 0.2 * scene_height)
            if intersection < 0.15:
                z_points.append(float('inf'))
                continue
                
            z_points.append(min_distances.min())
            
        z_points = np.array(z_points)
        object_height = obj.points[:, 2].max() - obj.points[:, 2].min()
        
        if np.all(np.isinf(z_points)):
            self._handle_no_support_found(obj, walls, scene_height, object_height)
        else:
            self._handle_support_found(obj, z_points, object_height)
            
    def _handle_no_support_found(self, obj, walls, scene_height, object_height):
        """Handle case where no support surface is found"""
        intersections = [
            obj.get_intersection(obj_, object_height + 0.5)
            for obj_ in [self.handler.objects[wall_id] for wall_id in self.handler.wall_ids]
        ]
        max_int = max(intersections)
        wall_normals = self.handler.objects[
            self.handler.wall_ids[intersections.index(max_int)]
        ].dominant_normal
        wall_normals = wall_normals / np.linalg.norm(wall_normals, keepdims=True, axis=-1)
        
        obj_normals = obj.dominant_normal
        obj_normals = obj_normals / np.linalg.norm(obj_normals, keepdims=True, axis=-1)
        
        sim_norm = obj_normals @ wall_normals.T
        if max_int > 0.3 and sim_norm > 0.99:
            obj.constraints.append([
                'hanging on', 'wall',
                self.handler.wall_ids[intersections.index(max_int)]
            ])
            print(f"{obj.obj_name} ({obj.obj_id}) hanging on wall "
                  f"({self.handler.wall_ids[intersections.index(max_int)]})")
        else:
            intersections = [
                obj.get_intersection(obj_, 0.2 * scene_height)
                if obj_ != obj and obj_.obj_name != 'wall'
                else 0.0
                for obj_ in self.handler.objects
            ]
            max_int = max(intersections)
            position = ('on top of' if (obj.points[:, 2].min() -
                       self.handler.objects[intersections.index(max_int)].points[:, 2].max()) > -0.1
                       else 'inside')
            obj.constraints.append([
                position,
                self.handler.class_names[intersections.index(max_int)],
                intersections.index(max_int)
            ])
            print(f"{obj.obj_name} ({obj.obj_id}) {position} "
                  f"{self.handler.class_names[intersections.index(max_int)]} "
                  f"({intersections.index(max_int)})")
                  
    def _handle_support_found(self, obj, z_points, object_height):
        """Handle case where support surface is found"""
        intersections = np.array([
            obj.get_intersection(obj_, object_height + 0.05)
            if obj_ != obj and obj_.obj_name != 'wall'
            else 0.0
            for obj_ in self.handler.objects
        ])
        
        distance_to_top = np.array([
            np.linalg.norm(
                obj.points[obj.points[:, 2] < obj.points[:, 2].min() + 0.05].mean(axis=0) -
                SurfaceUtils.get_top_support_surfaces_as_pointcloud(obj_.points),
                axis=-1
            ).min()
            if ((obj_ != obj) and (obj_.obj_name != 'corrupt') and
                ((intersections[obj_.obj_id]) != 0))
            else float('inf')
            for obj_ in self.handler.objects
        ])
        
        closest_3_indices = np.argsort(distance_to_top)[:3]
        
        if (self.handler.floor_id in closest_3_indices and
            intersections[self.handler.floor_id] > 0.7):
            closest_obj = self.handler.floor_id
        elif intersections[closest_3_indices].min() != 1.:
            closest_obj = closest_3_indices[np.argmax(intersections[closest_3_indices])]
        else:
            closest_obj = (closest_3_indices[0]
                          if distance_to_top[closest_3_indices[0]] < 0.5
                          else self.handler.floor_id)
            
        position = ('on top of'
                   if (obj.points[:, 2].min() -
                       self.handler.objects[closest_obj].points[:, 2].max()) > -0.02
                   else 'inside')
        obj.constraints.append([
            position,
            self.handler.objects[closest_obj].obj_name,
            closest_obj
        ])
        print(f"{obj.obj_name} ({obj.obj_id}) {position} "
              f"{self.handler.objects[closest_obj].obj_name} ({closest_obj})")
              
    def _get_floor_objects(self):
        """Get all objects on floor"""
        floor_objects = []
        obj_sizes = []
        for obj in self.handler.objects:
            if obj.obj_name == 'wall' or 'cc' in obj.obj_name:
                continue
            if obj.obj_name == 'corrupt':
                continue
            for c in obj.constraints:
                if 'floor' in c:
                    floor_objects.append(obj)
                    obj_sizes.append(obj.volume)
                    break
        return floor_objects, obj_sizes
        
    def _sort_floor_objects(self, floor_objects_data):
        """Sort floor objects by size"""
        floor_objects, obj_sizes = floor_objects_data
        return [obj for _, obj in sorted(
            zip(obj_sizes, floor_objects),
            key=lambda x: x[0],
            reverse=True
        )]
    def _collect_supported_objects(self):
        """Collect objects supported by each object"""
        collected_objects = [[] for _ in range(len(self.handler.objects))]
        for obj_id, obj in enumerate(self.handler.objects):
            if obj.obj_name == 'corrupt':
                continue
            for obj_id_, obj_ in enumerate(self.handler.objects):
                if obj_.obj_name == 'corrupt':
                    continue
                if obj_id == obj_id_:
                    continue
                for c in obj_.constraints:
                    if obj_id in c:
                        collected_objects[obj_id].append(obj_id_)
        return collected_objects
        
    def _merge_collected_objects(self, collected_objects, walls, scene_height):
        """Merge collected objects based on intersections"""
        collected_objects_merged = copy.deepcopy(collected_objects)
        for id, c_objs in enumerate(collected_objects):
            if self.handler.objects[id].obj_name == 'corrupt':
                continue
            for id_, c_objs_ in enumerate(collected_objects):
                if id == id_:
                    continue
                if self.handler.objects[id_].obj_name == 'corrupt':
                    continue
                if id_ in c_objs:
                    for id__ in c_objs_:
                        if self.handler.objects[id__].obj_name == 'corrupt':
                            continue
                        wall_intersection = sum([
                            self.handler.objects[id__].get_intersection(wall, scene_height)
                            for wall in walls
                        ])
                        obj_intersection = self.handler.objects[id__].get_intersection(
                            self.handler.objects[id], scene_height
                        )
                        if wall_intersection < 0.15 and obj_intersection > 0.15:
                            collected_objects_merged[id].append(id__)
        return collected_objects_merged
        
    def _set_object_surfaces(self, collected_objects_merged):
        """Set surfaces for objects"""
        def get_surface_elevation(s_elevations, obj):
            if len(s_elevations) == 0:
                return obj.points[:, 2].min()
            min_point = obj.points[:, 2].min()
            return s_elevations[np.argmin(np.linalg.norm(s_elevations - min_point))]
            
        for obj in self.handler.objects:
            if obj.obj_name == 'floor':
                continue
            if obj.obj_name == 'corrupt':
                continue
            if len(obj.surfaces) == 0:
                continue
                
            s_elevations = [s['elevation'][-1].item() for s in obj.surfaces]
            projected_points = [
                np.concatenate([
                    obj_.points[:, :2],
                    np.ones((len(obj_.points), 1)) * get_surface_elevation(s_elevations, obj_)
                ], axis=-1)
                for obj_ in [self.handler.objects[i] for i in collected_objects_merged[obj.obj_id]]
            ]
            projected_points += [
                np.concatenate([
                    obj.points[:, :2][obj.points[:, 2] > s_e - 0.1],
                    np.ones((len(obj.points[:, :2][obj.points[:, 2] > s_e - 0.1]), 1)) * s_e
                ], axis=-1)
                for s_e in s_elevations
            ]
            
            if len(projected_points) > 0:
                projected_points = np.concatenate(projected_points)
                min_box = obj.points.min(axis=0)
                max_box = obj.points.max(axis=0)
                projected_points = projected_points[
                    ((projected_points[:, 0] < max_box[0]) *
                     (projected_points[:, 0] > min_box[0]) *
                     (projected_points[:, 1] < max_box[1]) *
                     (projected_points[:, 1] > min_box[1])) == 1
                ]
                if len(projected_points) > 0:
                    obj.set_surfaces(projected_supported_objects=projected_points)
                    
    def _save_objects(self):
        """Save objects to file"""
        import os
        dir_save = self.handler.save_in
        os.makedirs(dir_save, exist_ok=True)
        
        extension = '_preds' if self.handler.use_preds else ''
        objects_list = []
        for obj in self.handler.objects:
            objects_list.append(obj.construct_dictionary())
            
        chunk_size = 100
        for i in range(0, len(objects_list), chunk_size):
            chunk = objects_list[i : i + chunk_size]
            torch.save(
                chunk,
                os.path.join(self.handler.save_in, f'objects{extension}_{i//chunk_size}.pt')
            )
            
    def get_scene_graph(self):
        """Get scene graph representation"""
        scene_graph = []
        for obj_id, obj in enumerate(self.handler.objects):
            if obj.obj_name == 'corrupt':
                continue
            scene_graph.append({
                "object name": obj.obj_name,
                "object id": obj_id,
                "Semantic constraints": obj.constraints,
                "region in image": obj.image_constraints,
                'coordinates in image': obj.xy_coords,
                'projected points': obj.projected_points,
                'projected dominant normal': obj.projected_dominant_normal
            })
        return scene_graph