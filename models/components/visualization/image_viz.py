"""
2D Image Visualization

Functions for drawing on images: arrows, bounding boxes, movement visualization.
Used for showing object movements and 2D scene representations.
"""

import math
import random
import copy
from PIL import Image, ImageDraw, ImageFont


def draw_bboxes_from_camera(objects, obj_ids, cameras, image):
    """
    Draw 2D bounding boxes with labels on image from camera view.
    
    Projects 3D object bounding boxes to 2D image space and draws them
    with unique colors and ID labels.
    
    Args:
        objects (list): List of objects with get_2d_bbox() method
        obj_ids (list): Object IDs to visualize
        cameras: Camera object(s) for projection
        image (PIL.Image): Base image to draw on
    
    Returns:
        PIL.Image: Image with bounding boxes and labels drawn
        
    Example:
        >>> img = Image.open("scene.png")
        >>> result = draw_bboxes_from_camera(
        ...     scene.objects, [0, 1, 2], camera, img
        ... )
        >>> result.save("annotated.png")
        
    Use case:
        Visualizing object detections and scene understanding results
    """
    from utils.color_utils import generate_color_dict
    
    # Generate unique colors for each object
    colors = generate_color_dict(len(obj_ids))
    
    # Get 2D bounding boxes for all objects
    bboxes = [objects[i].get_2d_bbox(cameras) for i in obj_ids]
    
    # Draw on image
    draw = ImageDraw.Draw(image)
    
    for obj_id, bbox in zip(obj_ids, bboxes):
        # Get color for this object
        color_name = list(colors.keys())[obj_ids.index(obj_id)]
        color_rgb = tuple(int(c * 255) for c in colors[color_name])
        
        # Draw bounding box rectangle
        draw.rectangle(bbox, outline=color_rgb, width=3)
        
        # Add label above box
        label_position = (bbox[0], bbox[1] - 10)
        draw.text(label_position, f"ID: {obj_id}", fill=color_rgb)
    
    return image

def visualize_movement(image, initialization, scene_graph, arrow_width=5, font_size=20):
    """
    Visualize object movements as arrows on an image.
    
    Draws colored arrows from initial positions to target positions,
    with object names labeled and movement magnitude indicated by arrow length.
    
    Args:
        image (PIL.Image): Base image to draw on
        initialization (list): Target positions [(x, y), ...]
        scene_graph (list of dict): Scene objects with 'coordinates in image' and 'object name'
        arrow_width (int): Width of arrow lines. Default: 5
        font_size (int): Size of text labels. Default: 20
    
    Returns:
        PIL.Image: Image with movement arrows drawn
        
    Example:
        >>> img = Image.open("scene.png")
        >>> init_pos = [(100, 100), (200, 150)]
        >>> scene = [{'coordinates in image': {'x': 50, 'y': 50}, 'object name': 'chair'}, ...]
        >>> result = visualize_movement(img, init_pos, scene)
    """
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    end_points = []
    start_points = []
    obj_names = []
    
    # Load font
    try:
        arrow_font = ImageFont.truetype("./fonts/Roboto-Bold.ttf", font_size)
        magnitude_font = ImageFont.truetype("./fonts/Roboto-Bold.ttf", font_size)
    except IOError:
        arrow_font = ImageFont.load_default()
        magnitude_font = ImageFont.load_default()

    # Extract start and end points
    for o_id, init_ in enumerate(initialization):
        if init_ is not None:
            end_points.append(tuple(init_))
            start_points.append((
                scene_graph[o_id]['coordinates in image']['x'],
                scene_graph[o_id]['coordinates in image']['y']
            ))
            obj_names.append(scene_graph[o_id]['object name'])

    if len(start_points) != len(end_points):
        raise ValueError("Start points and end points must have the same length.")

    # Draw each arrow
    for o_id, (start, end) in enumerate(zip(start_points, end_points)):
        # Random color for this object
        color = tuple(random.randint(0, 255) for _ in range(3))
        darker_color = tuple(min(255, int(c * 1.5)) for c in color)

        # Draw arrow line
        draw.line([start, end], fill=color, width=arrow_width)

        # Calculate magnitude
        magnitude = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

        # Middle point for label
        mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)

        # Draw arrowhead
        arrow_tip_length = arrow_width * 3
        dx, dy = end[0] - start[0], end[1] - start[1]

        point1 = (
            end[0] - arrow_tip_length * (dx - dy) / ((dx**2 + dy**2)**0.5),
            end[1] - arrow_tip_length * (dy + dx) / ((dx**2 + dy**2)**0.5),
        )
        point2 = (
            end[0] - arrow_tip_length * (dx + dy) / ((dx**2 + dy**2)**0.5),
            end[1] - arrow_tip_length * (dy - dx) / ((dx**2 + dy**2)**0.5),
        )
        draw.polygon([end, point1, point2], fill=color)

        # Add object name as label
        text_position = (mid_point[0] + 5, mid_point[1] - 5)
        draw.text(text_position, f"{obj_names[o_id]}", fill=darker_color, font=magnitude_font)

    return image_copy


def draw_bounding_box(img_, distance_to_obj_ratio=1.0):
    """
    Draw bounding box on image based on distance ratio.
    
    Creates a rectangular box centered in the image, with size
    determined by distance_to_obj_ratio parameter.
    
    Args:
        img_ (PIL.Image): Input image
        distance_to_obj_ratio (float): Ratio for box size. Default: 1.0
    
    Returns:
        PIL.Image: Image with bounding box drawn
        
    Example:
        >>> img = Image.open("photo.png")
        >>> result = draw_bounding_box(img, distance_to_obj_ratio=0.5)
        >>> # Box covers center 50% of image
    """
    img = copy.deepcopy(img_)
    
    # Calculate box coordinates
    box_coords = (
        ((int(4) * distance_to_obj_ratio) // 2 - 1) * img.size[0] // (int(4) * distance_to_obj_ratio),
        ((int(4) * distance_to_obj_ratio) // 2 - 1) * img.size[1] // (int(4) * distance_to_obj_ratio),
        ((int(4) * distance_to_obj_ratio) // 2 + 1) * img.size[0] // (int(4) * distance_to_obj_ratio),
        ((int(4) * distance_to_obj_ratio) // 2 + 1) * img.size[1] // (int(4) * distance_to_obj_ratio)
    )
    
    # Draw rectangle
    draw = ImageDraw.Draw(img)
    draw.rectangle(box_coords, outline="red", width=2)
    
    return img