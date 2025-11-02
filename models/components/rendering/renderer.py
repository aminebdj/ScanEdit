import numpy as np
import torch
import copy
from PIL import Image
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardPhongShader, BlendParams,
    PerspectiveCameras, AmbientLights, PointLights, RasterizationSettings
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.renderer.cameras import look_at_view_transform
from models.components.utils.geometry_utils import rotation_matrix_from_vector_to_z


class SceneRenderer:
    """Handles all rendering operations for the scene"""
    
    def __init__(self, handler):
        self.handler = handler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def render_object(self, render_only, focal_length=2.0, width=1920, height=1080,
                     render_front=True, incline_front=True, pad=True,
                     keep_other_objects=False, look_at_point=None, look_from=None,
                     use_dummy_data=None, pad_percent=0.10, get_depth=False,
                     add_half_height=True, get_original_image=False):
        """
        Render specific objects from the scene
        
        Args:
            render_only: List of object IDs to render
            focal_length: Camera focal length
            width: Image width
            height: Image height
            render_front: Whether to render from front
            incline_front: Whether to incline camera when rendering front
            pad: Whether to pad the cropped image
            keep_other_objects: Whether to keep other objects in scene
            look_at_point: Optional point to look at
            look_from: Optional camera position
            use_dummy_data: Index of dummy data to use
            pad_percent: Padding percentage
            get_depth: Whether to return depth map
            add_half_height: Whether to add half height to camera position
            get_original_image: Whether to return original uncropped image
            
        Returns:
            Tuple of (image, min_point, max_point, width, height, cameras)
        """
        data = self.handler.data_dummy[use_dummy_data] if use_dummy_data is not None else self.handler.data
        
        # Extract mesh data
        vertices = copy.deepcopy(np.asarray(data.vertices))
        faces = copy.deepcopy(np.asarray(data.triangles))
        vertex_normals = copy.deepcopy(np.asarray(data.vertex_normals))
        vertex_colors = copy.deepcopy(np.asarray(data.vertex_colors))
        
        # Create mask for rendering
        point_indx = np.arange(len(vertices))
        mask_central_obj = self.handler.class_agnostic_masks[render_only[0]]
        mask = sum(self.handler.class_agnostic_masks[object_id] for object_id in render_only) == 1
        
        if not keep_other_objects:
            mask_cut = mask
        else:
            mask_cut = ~(sum(self.handler.class_agnostic_masks[object_id] 
                           for object_id in [id for id, cn in enumerate(self.handler.class_names) 
                                           if cn == 'wall']) == 1)
            mask_cut = ~self.handler.room_rendering_mask
            
        # Filter faces and vertices
        inst_indices = point_indx[mask_cut]
        incl_mask = np.isin(faces, inst_indices).sum(axis=-1)
        incl_mask = ((incl_mask == 1) + (incl_mask == 2)) >= 1
        faces = faces[~incl_mask]
        vertices[~mask_cut] = self.handler.discard_point
        
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            raise ValueError("Vertices or faces are empty after masking.")
            
        # Convert to PyTorch tensors
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)
        faces_tensor = torch.from_numpy(faces).long().to(self.device)
        vertex_colors_tensor = torch.from_numpy(vertex_colors).float().to(self.device)
        vertex_normals_tensor = torch.from_numpy(vertex_normals).float().to(self.device)
        
        # Create mesh
        textures = Textures(verts_rgb=[vertex_colors_tensor])
        object_mesh = Meshes(
            verts=[vertices_tensor],
            faces=[faces_tensor],
            textures=textures
        )
        
        meshes_list = [object_mesh]
        
        # Add floor mesh if available
        if self.handler.floor_mesh is not None:
            floor_mesh = self._create_floor_mesh()
            meshes_list.append(floor_mesh)
            
        # Combine meshes
        mesh = join_meshes_as_scene(meshes_list)
        
        # Setup camera
        center = vertices[mask_central_obj].mean(axis=0) if look_at_point is None else look_at_point
        front_normal = np.array(self.handler.objects[render_only[0]].dominant_normal)
        front_normal /= np.linalg.norm(front_normal)
        extents = vertices[mask_central_obj].ptp(axis=0)
        face_area = max(extents[0] * extents[1], extents[1] * extents[2], extents[0] * extents[2])
        height_obj = vertices[mask_central_obj][:, 2].max() - vertices[mask_central_obj][:, 2].min()
        
        # Calculate camera position
        camera_position = self._calculate_camera_position(
            center, front_normal, face_area, height_obj, 
            render_front, incline_front, look_from, add_half_height, 
            mask_central_obj, vertices
        )
        
        # Setup lighting
        point_light_loc = self._calculate_light_position(
            center, front_normal, face_area, height_obj,
            render_front, incline_front, camera_position
        )
        
        lights = self._setup_lights(point_light_loc)
        
        # Setup camera transform
        R, T = look_at_view_transform(
            eye=torch.tensor(camera_position, dtype=torch.float).to(self.device)[None, :],
            at=torch.tensor(center, dtype=torch.float).to(self.device)[None, :],
            up=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float).to(self.device)[None, :]
        )
        
        cameras = PerspectiveCameras(
            R=R, T=T, device=self.device, focal_length=focal_length,
            image_size=torch.tensor([[height, width]]).to(self.device)
        )
        
        # Render
        renderer = self._create_renderer(cameras, height, width, lights)
        
        if get_depth:
            return self._render_depth(renderer, mesh)
            
        rendered_image = renderer(mesh)
        rendered_image = rendered_image[0, ..., :3].cpu().numpy()
        image = Image.fromarray((rendered_image * 255).astype(np.uint8))
        
        if look_at_point is not None:
            return image, np.array([0, 0, 0]), None, None, None, cameras
            
        if get_original_image:
            return image
            
        # Crop image
        return self._crop_rendered_image(image, cameras, vertices, mask, width, height, pad, pad_percent)
        
    def _create_floor_mesh(self):
        """Create floor mesh for rendering"""
        floor_vertices = torch.from_numpy(np.asarray(self.handler.floor_mesh.vertices)).float().to(self.device)
        floor_faces = torch.from_numpy(np.asarray(self.handler.floor_mesh.triangles)).long().to(self.device)
        floor_colors = torch.from_numpy(np.asarray(self.handler.floor_mesh.vertex_colors)).float().to(self.device)
        floor_normals = torch.from_numpy(np.asarray(self.handler.floor_mesh.vertex_normals)).float().to(self.device)
        floor_textures = Textures(verts_rgb=[floor_colors])
        
        return Meshes(
            verts=[floor_vertices],
            faces=[floor_faces],
            textures=floor_textures,
            verts_normals=[floor_normals]
        )
        
    def _calculate_camera_position(self, center, front_normal, face_area, height_obj,
                                   render_front, incline_front, look_from, 
                                   add_half_height, mask_central_obj, vertices):
        """Calculate camera position based on rendering parameters"""
        if render_front and incline_front:
            camera_position = center + front_normal * 2 * face_area + np.array([0, 0, height_obj / 1.2])
        elif render_front:
            camera_position = center + front_normal * 2 * face_area
            if add_half_height:
                obj_v = vertices[mask_central_obj]
                obj_dims = obj_v.max(axis=0) - obj_v.min(axis=0)
                obj_dims[0] = 0
                obj_dims[1] = 0
                camera_position += obj_dims / 2
        else:
            camera_position = center + np.array([0, (1 - 0.999**2)**(0.5), 0.999]) * 2 * face_area
        if look_from is not None:
            camera_position = look_from
            
        return camera_position
        
    def _calculate_light_position(self, center, front_normal, face_area, height_obj,
                                  render_front, incline_front, camera_position):
        """Calculate light position based on rendering parameters"""
        if render_front and incline_front:
            return (center + front_normal * face_area + np.array([0, 0, height_obj / 1.2])).tolist()
        elif render_front:
            return (center + front_normal * face_area).tolist()
        else:
            return camera_position.tolist()
            
    def _setup_lights(self, point_light_loc):
        """Setup scene lighting"""
        light_params = []
        for ambient_intensity in [1.5]:
            diffuse_intensity = 5.0
            specular_intensity = 1.0
            
            total_intensity = ambient_intensity + diffuse_intensity + specular_intensity
            ambient_intensity /= total_intensity
            diffuse_intensity /= total_intensity
            specular_intensity /= total_intensity
            
            light_params.append([ambient_intensity, diffuse_intensity, specular_intensity])
            
        return PointLights(
            location=[point_light_loc for _ in light_params],
            ambient_color=tuple([tuple([lp[0], lp[0], lp[0]]) for lp in light_params]),
            diffuse_color=tuple([tuple([lp[1], lp[1], lp[1]]) for lp in light_params]),
            specular_color=tuple([tuple([lp[2], lp[2], lp[2]]) for lp in light_params]),
            device=self.device
        )
        
    def _create_renderer(self, cameras, height, width, lights):
        """Create PyTorch3D renderer"""
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        
        return MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )
        
    def _render_depth(self, renderer, mesh):
        """Render depth map"""
        fragments = renderer.rasterizer(mesh)
        depth = fragments.zbuf.squeeze().cpu().numpy()
        depth[~np.isfinite(depth)] = 0
        return depth
        
    def _crop_rendered_image(self, image, cameras, vertices, mask, width, height, pad, pad_percent):
        """Crop rendered image to object bounds"""
        projected_points = cameras.transform_points_screen(
            torch.from_numpy(vertices[mask]).float().cuda()
        )
        dims = projected_points.max(dim=0).values[:2] - projected_points.min(dim=0).values[:2]
        h_w_10p = (dims) * pad_percent
        
        if pad:
            max_point = (projected_points[:, :2].max(axis=0).values + h_w_10p).tolist()
            min_point = (projected_points[:, :2].min(axis=0).values - h_w_10p).tolist()
        else:
            max_point = projected_points[:, :2].max(axis=0).values.tolist()
            min_point = projected_points[:, :2].min(axis=0).values.tolist()
            
        # Clamp to image bounds
        min_point = np.maximum(min_point, [0, 0])
        max_point = np.minimum(max_point, [width, height])
        
        # Crop
        min_x, min_y = map(int, min_point)
        max_x, max_y = map(int, max_point)
        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        
        return cropped_image, min_point, max_point, width, height, cameras
        
    def render_room(self, output_dir="./output/rendered_global_frame", width=1920, height=1080,
                   distance_to_obj_ratio=2.0, focal_length=2.0, render_only=None,
                   use_lights_shadows=True, use_dummy_data=None, get_depth=False, region=None):
        """
        Render entire room from top view
        
        Args:
            output_dir: Directory to save output
            width: Image width
            height: Image height
            distance_to_obj_ratio: Camera distance ratio
            focal_length: Camera focal length
            render_only: Optional list of object groups to render only
            use_lights_shadows: Whether to use lighting and shadows
            use_dummy_data: Index of dummy data to use
            get_depth: Whether to return depth map
            region: Optional region specification
            
        Returns:
            Tuple of (image, min_point, max_point, width, height, cameras)
        """
        import os
        # os.makedirs(output_dir, exist_ok=True)
        
        data = self.handler.data_dummy[use_dummy_data] if use_dummy_data is not None else self.handler.data
        
        if data.is_empty():
            return None
            
        # Extract mesh data
        vertices = np.asarray(data.vertices)
        faces = np.asarray(data.triangles)
        vertex_normals = np.asarray(data.vertex_normals)
        vertex_colors = np.asarray(data.vertex_colors)
        
        # Create mask for rendering
        point_indx = np.arange(len(vertices))
        mask = copy.deepcopy(self.handler.room_rendering_mask)
        
        if render_only is not None:
            all_ids = []
            for list_ in render_only:
                all_ids += list_
            render_masks = sum([self.handler.class_agnostic_masks[id] for id in all_ids]) == 1
            mask += ~render_masks
            mask = mask == 1
            
        # Filter faces
        inst_indices = point_indx[mask]
        incl_mask = np.isin(faces, inst_indices).sum(axis=-1)
        incl_mask = ((incl_mask == 1) + (incl_mask == 2)) >= 1
        faces = faces[~incl_mask]
        
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            return None
            
        # Convert to tensors
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device)
        faces_tensor = torch.from_numpy(faces).long().to(self.device)
        vertex_normals_tensor = torch.from_numpy(vertex_normals).float().to(self.device)
        vertex_colors_tensor = torch.from_numpy(vertex_colors).float().to(self.device)
        
        # Mask out room rendering areas
        vertices_tensor[self.handler.room_rendering_mask] = float("-inf")
        
        if render_only is not None:
            all_ids = []
            for list_ in render_only:
                all_ids += list_
            render_masks = sum([self.handler.class_agnostic_masks[id] for id in all_ids]) == 1
            vertices_tensor[~render_masks] = float("-inf")
            
        # Mask floor if floor mesh exists
        if self.handler.floor_mesh is not None:
            vertices_tensor[self.handler.class_agnostic_masks[self.handler.floor_id]] = float("-inf")
            
        # Create mesh
        textures = Textures(verts_rgb=[vertex_colors_tensor])
        mesh = Meshes(
            verts=[vertices_tensor],
            faces=[faces_tensor],
            textures=textures,
            verts_normals=[vertex_normals_tensor]
        )
        
        meshes_list = [mesh]
        
        # Add floor mesh
        if self.handler.floor_mesh is not None:
            floor_mesh = self._create_floor_mesh()
            meshes_list.append(floor_mesh)
            
        mesh = join_meshes_as_scene(meshes_list)
        
        # Setup lighting
        point_light_loc = (vertices.max(axis=0) + 1).tolist()
        
        if use_lights_shadows:
            lights = self._setup_lights(point_light_loc)
        else:
            lights = AmbientLights(device=self.device)
            
        # Calculate camera position
        masked_indices = np.where(~self.handler.room_rendering_mask)[0]
        masked_points = vertices[masked_indices]
        
        R_wall = rotation_matrix_from_vector_to_z(
            self.handler.wall_normal,
            target=np.array([0, 1, 0])
        )
        rotated_points = R_wall @ masked_points.T
        camera_distance = np.linalg.norm(
            np.max(rotated_points, axis=1) - np.min(rotated_points, axis=1)
        ) * 1.0
        center_np = np.mean(rotated_points, axis=-1)
        camera_pos = center_np + camera_distance * self.handler.floor_normal
        
        # Setup camera
        R, T = look_at_view_transform(
            eye=camera_pos[None, :],
            at=center_np[None, :],
            up=torch.tensor([0.0, 1.0, 0.0])[None, :].to(self.device)
        )
        R = R @ R_wall
        
        cameras = PerspectiveCameras(
            R=R.to(self.device),
            T=T.to(self.device),
            device=self.device,
            focal_length=focal_length,
            image_size=torch.tensor([[height, width]]).to(self.device)
        )
        
        # Render
        renderer = self._create_renderer(cameras, height, width, lights)
        
        if get_depth:
            return self._render_depth(renderer, mesh)
            
        rendered_image = renderer(mesh, cameras=cameras)
        rendered_image = rendered_image[0].cpu().numpy()
        color = rendered_image[..., :3]
        
        # Calculate crop region
        projected_points = cameras.transform_points_screen(
            torch.from_numpy(masked_points).float().cuda()
        )
        max_point = projected_points[:, :2].max(axis=0).values.tolist()
        min_point = projected_points[:, :2].min(axis=0).values.tolist()
        
        image = Image.fromarray((color * 255).astype(np.uint8))
        
        min_x, min_y = map(int, min_point)
        max_x, max_y = map(int, max_point)
        
        if region is not None:
            height = max_y - min_y
            width = max_x - min_x
            min_x += width // 3
            max_x -= width // 3
            min_y += height // 3
            max_y -= height // 3
            min_point = [min_x, min_y]
            max_point = [max_x, max_y]
            
        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        
        # Store floor points for later use
        if self.handler.floor_mesh is not None:
            floor_vertices = torch.from_numpy(
                np.asarray(self.handler.floor_mesh.vertices)
            ).float().to(self.device)
            self.handler.floor_points = (
                torch.unique(floor_vertices // 0.05, dim=0) * 0.05
            )
            self.handler.proj_2d = cameras.transform_points_screen(
                self.handler.floor_points
            ).cpu().numpy()[:, :2] - min_point
            self.handler.floor_points = self.handler.floor_points.cpu().numpy()
            
        return cropped_image, min_point, max_point, max_x - min_x, max_y - min_y, cameras
        
    def render_depth(self, target_point, target_normal, width=1920, height=1080,
                    distance_to_obj_ratio=2.0, focal_length=2.0, exclude_masks=None):
        """
        Render depth map from specific viewpoint
        
        Args:
            target_point: Point to look at
            target_normal: Normal vector at target point
            width: Image width
            height: Image height
            distance_to_obj_ratio: Camera distance ratio
            focal_length: Camera focal length
            exclude_masks: Optional masks to exclude
            
        Returns:
            Tuple of (depth_image, cameras)
        """
        if self.handler.data.is_empty():
            return None
            
        vertices = np.asarray(self.handler.data.vertices)
        faces = np.asarray(self.handler.data.triangles)
        vertex_colors = np.asarray(self.handler.data.vertex_colors)
        
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device).unsqueeze(0)
        
        if exclude_masks is not None:
            vertices_tensor[:, ~exclude_masks] = float("-inf")
            
        faces_tensor = torch.from_numpy(faces).long().to(self.device).unsqueeze(0)
        vertex_colors_tensor = torch.from_numpy(vertex_colors).float().to(self.device).unsqueeze(0)
        
        textures = Textures(verts_rgb=vertex_colors_tensor)
        mesh = Meshes(verts=vertices_tensor, faces=faces_tensor, textures=textures)
        
        target_point = torch.tensor(target_point, dtype=torch.float32).to(self.device)
        target_normal = torch.tensor(target_normal, dtype=torch.float32).to(self.device)
        
        camera_pos = target_point + 0.2 * torch.nn.functional.normalize(target_normal, dim=0)
        up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).to(self.device)
        
        R, T = look_at_view_transform(
            eye=camera_pos.squeeze().unsqueeze(0),
            at=target_point.squeeze().unsqueeze(0),
            up=up
        )
        
        cameras = PerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            focal_length=torch.tensor([[focal_length, focal_length]]).to(self.device),
            image_size=torch.tensor([[height, width]]).to(self.device)
        )
        
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=self.device, cameras=cameras)
        )
        
        fragments = renderer.rasterizer(mesh)
        depth = fragments.zbuf.squeeze().cpu().numpy()
        depth[~np.isfinite(depth)] = 0
        
        # Normalize depth
        depth_min = depth[depth > 0].min() if np.any(depth > 0) else 0
        depth_max = depth.max()
        neg_mask = depth < 0
        
        if depth_max > depth_min:
            depth_normalized = 255 - ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = 255 - (depth * 255).astype(np.uint8)
            
        depth_normalized[neg_mask] = 0
        non_face_mask = depth == 0
        depth_normalized[non_face_mask] = 0
        
        depth_image = Image.fromarray(depth_normalized, mode='L')
        
        return depth_image, cameras
        
    def render_frames(self, mask, k, angles=[-np.pi / 6 + np.pi / 2, np.pi / 2, np.pi / 6 + np.pi / 2],
                     render_background=True, output_dir="./output/rendered_frames",
                     width=1920, height=1080, distance_to_obj_ratio=1.5, m_id=0):
        """
        Render multiple frames from different angles
        
        Args:
            mask: Mask for object to render
            k: Number of frames per angle
            angles: List of phi angles
            render_background: Whether to render background
            output_dir: Output directory
            width: Image width
            height: Image height
            distance_to_obj_ratio: Camera distance ratio
            m_id: Object ID
            
        Returns:
            List of rendered frames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.handler.data.is_empty():
            return []
            
        vertices = np.asarray(self.handler.data.vertices)
        faces = np.asarray(self.handler.data.triangles)
        vertex_normals = np.asarray(self.handler.data.vertex_normals)
        vertex_colors = np.asarray(self.handler.data.vertex_colors)
        
        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            return []
            
        # Convert to tensors
        vertices_tensor = torch.from_numpy(vertices).float().to(self.device).unsqueeze(0)
        vertices_tensor_no_bg = torch.from_numpy(vertices).float().to(self.device).unsqueeze(0)
        vertices_tensor_no_bg[:, ~mask] = float("-inf")
        
        if not render_background:
            vertices_tensor[:, ~mask] = float("-inf")
            
        faces_tensor = torch.from_numpy(faces).long().to(self.device).unsqueeze(0)
        vertex_normals_tensor = torch.from_numpy(vertex_normals).float().to(self.device).unsqueeze(0)
        vertex_colors_tensor = torch.from_numpy(vertex_colors).float().to(self.device).unsqueeze(0)
        
        textures = Textures(verts_rgb=vertex_colors_tensor)
        
        mesh = Meshes(
            verts=vertices_tensor,
            faces=faces_tensor,
            textures=textures,
            verts_normals=vertex_normals_tensor
        )
        mesh_no_bg = Meshes(
            verts=vertices_tensor_no_bg,
            faces=faces_tensor,
            textures=textures,
            verts_normals=vertex_normals_tensor
        )
        
        # Calculate center and camera distance
        masked_indices = np.where(mask)[0]
        masked_points = vertices[masked_indices]
        center_np = np.mean(masked_points, axis=0)
        camera_distance = np.linalg.norm(
            np.max(masked_points, axis=0) - np.min(masked_points, axis=0)
        ) * distance_to_obj_ratio
        
        lights = AmbientLights(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1
        )
        background_color = (1.0, 1.0, 1.0)
        
        rendered_frames = []
        
        for j, phi in enumerate(angles):
            for i in range(k):
                theta = 2 * np.pi * i / k
                
                # Calculate camera position
                camera_pos = center_np + camera_distance * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                
                R, T = look_at_view_transform(
                    eye=camera_pos[None, :],
                    at=center_np[None, :],
                    up=torch.tensor([0.0, 0.0, 1.0])[None, :].to(self.device)
                )
                
                cameras = PerspectiveCameras(
                    R=R.to(self.device),
                    T=T.to(self.device),
                    device=self.device,
                    focal_length=2,
                    image_size=torch.tensor([[height, width]]).to(self.device)
                )
                
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                    shader=HardPhongShader(
                        blend_params=BlendParams(background_color=background_color),
                        device=self.device,
                        cameras=cameras,
                        lights=lights
                    )
                )
                
                # Render depth maps
                fragments = renderer.rasterizer(mesh, cameras=cameras)
                fragments_no_bg = renderer.rasterizer(mesh_no_bg, cameras=cameras)
                
                depth = fragments.zbuf.squeeze()
                depth_numpy = depth.cpu().numpy()
                
                depth_no_bg = fragments_no_bg.zbuf.squeeze()
                depth_numpy_no_bg = depth_no_bg.cpu().numpy()
                
                # Render image
                rendered_image = renderer(mesh, cameras=cameras)
                rendered_image = rendered_image[0].cpu().numpy()
                color = rendered_image[..., :3]
                
                image = Image.fromarray((color * 255).astype(np.uint8))
                
                # Check occlusion
                if ((depth_no_bg > depth) * (depth_no_bg != -1)).sum() / (depth_no_bg != -1).sum() > 0.1:
                    continue
                    
                rendered_frames.append(image)
                
                # Save image and depth
                os.makedirs(os.path.join(output_dir, str(distance_to_obj_ratio)), exist_ok=True)
                frame_path = os.path.join(
                    output_dir, str(distance_to_obj_ratio), f"frame_{m_id}_{j}_{i}.png"
                )
                image.save(frame_path)
                
                # Normalize and save depth
                depth_min = depth_numpy.min()
                depth_max = depth_numpy.max()
                depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min) * 255
                depth_normalized = depth_normalized.astype(np.uint8)
                depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
                depth_image.save(frame_path.replace('frame_', 'depth_'))
                
        return rendered_frames