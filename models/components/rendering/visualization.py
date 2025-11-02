import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont
from models.components.utils.color_utils import generate_color_dict


class Visualizer:
    """Handles visualization of bounding boxes and other visual elements"""
    
    def __init__(self, handler):
        self.handler = handler
        self.font_path = "/home/boudjoghra/projects/pc_pred/fonts/Roboto-Regular.ttf"
        
    def draw_bboxes_from_camera(self, obj_ids, cameras, image_og):
        """
        Draw bounding boxes for multiple objects
        
        Args:
            obj_ids: List of object IDs
            cameras: Camera object
            image_og: Original image
            
        Returns:
            Tuple of (image with boxes, color dict)
        """
        image = copy.deepcopy(image_og)
        colors = generate_color_dict(len(obj_ids))
        bboxes = [self.handler.objects[i].get_2d_bbox(cameras) for i in obj_ids]
        
        font = ImageFont.truetype(self.font_path, size=20)
        draw = ImageDraw.Draw(image)
        
        for obj_id, bbox in zip(obj_ids, bboxes):
            color_name = list(colors.keys())[obj_ids.index(obj_id)]
            color_rgb = tuple(int(c * 255) for c in colors[color_name])
            
            # Draw bbox
            draw.rectangle(bbox, outline=color_rgb, width=3)
            
            # Draw label
            label_position = (bbox[0], bbox[1] - 30)
            draw.text(
                label_position,
                f"{self.handler.objects[obj_id].obj_name}",
                fill=color_rgb,
                font=font
            )
            
        return image, colors
        
    def draw_group_bboxes_from_camera(self, obj_ids, cameras, image_og, color,
                                     draw_center=False, new_loc=None):
        """
        Draw bounding box for a group of objects
        
        Args:
            obj_ids: List of object IDs in group
            cameras: Camera object
            image_og: Original image
            color: RGB color tuple
            draw_center: Whether to draw center coordinates
            new_loc: Optional new location for bbox
            
        Returns:
            Image with group bbox drawn
        """
        image = copy.deepcopy(image_og)
        vertices = np.asarray(self.handler.data_dummy.vertices)
        points = [vertices[self.handler.class_agnostic_masks[id]][:2000] for id in obj_ids]
        bboxes = [self._get_2d_bbox(points_, cameras) for points_ in points]
        
        # Calculate overall bbox
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[2] for bbox in bboxes)
        max_y = max(bbox[3] for bbox in bboxes)
        bbox = (min_x, min_y, max_x, max_y)
        
        if new_loc is not None:
            bbox = self._shift_bbox(bbox, new_loc)
            
        font = ImageFont.truetype(self.font_path, size=20)
        draw = ImageDraw.Draw(image)
        color_rgb = tuple(int(c * 255) for c in color)
        
        # Draw bbox
        draw.rectangle(bbox, outline=color_rgb, width=6)
        
        # Draw center if requested
        if draw_center:
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_coords = (center_x, center_y)
            draw.text(center_coords, f"{center_coords}", fill=color_rgb, font=font)
            
        return image
        
    @staticmethod
    def _get_2d_bbox(points, cameras):
        """Calculate 2D bounding box from 3D points"""
        import torch
        camera = cameras[0]
        min_point_cropped = cameras[1]
        projected_points = camera.transform_points_screen(
            torch.from_numpy(points).cuda().float()
        ).cpu().numpy()
        min_point = projected_points[:, :2].min(axis=0) - min_point_cropped[:2]
        max_point = projected_points[:, :2].max(axis=0) - min_point_cropped[:2]
        return (int(min_point[0]), int(min_point[1]), int(max_point[0]), int(max_point[1]))
        
    @staticmethod
    def _shift_bbox(bbox, new_loc):
        """Shift bbox to align center with new location"""
        if new_loc is None:
            return bbox
            
        min_x, min_y, max_x, max_y = bbox
        new_x, new_y = new_loc
        
        current_x = (min_x + max_x) / 2
        current_y = (min_y + max_y) / 2
        
        shift_x = new_x - current_x
        shift_y = new_y - current_y
        
        return (
            min_x + shift_x,
            min_y + shift_y,
            max_x + shift_x,
            max_y + shift_y
        )