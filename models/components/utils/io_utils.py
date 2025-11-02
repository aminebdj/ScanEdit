import os
import torch
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path

def load_or_execute(root_path, filename, load_prev, function, *args, **kwargs):
    """
    A wrapper to check for a file and load it if it exists; otherwise, execute a given analysis function.

    Args:
        root_path (str): The root directory path.
        filename (str): The name of the file to check and load.
        analysis_function (callable): The function to execute if the file does not exist.
        *args: Positional arguments to pass to the analysis function.
        **kwargs: Keyword arguments to pass to the analysis function.

    Returns:
        Object: Loaded or newly created object.
    """
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, filename)
    if filename.endswith('pt'):
        if os.path.exists(file_path) and load_prev:
            return torch.load(file_path)
        else:
            result = function(*args, **kwargs)
            torch.save(result, file_path)
            return result
    elif filename.endswith('yaml'):
        if os.path.exists(file_path):
            return load_yaml(file_path)['answer']

        else:
            result = function(*args, **kwargs)
            data  = {
                'instruction' : args[0],
                'answer': result
            }
            save_yaml(data, file_path)
            return result

def load_yaml(file_path):
    """
    Load configuration from YAML file.
    
    Args:
        file_path (str or Path): Path to YAML file
    
    Returns:
        dict: Parsed YAML content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file contains invalid YAML
        
    Example:
        >>> config = load_yaml("config/scene_params.yaml")
        >>> print(config['optimization']['learning_rate'])
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def save_yaml(data, file_path):
    """
    Save data to YAML file.
    
    Args:
        data (dict): Data to save
        file_path (str or Path): Output file path
        
    Example:
        >>> config = {'learning_rate': 0.01, 'epochs': 100}
        >>> save_yaml(config, "config/training.yaml")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def save_points(points, path="./output/out.ply"):
    """
    Save numpy array of points to PLY file.
    
    Convenience function for quickly saving point clouds.
    
    Args:
        points (np.ndarray): Points to save, shape (N, 3)
        path (str): Output file path. Default: "./output/out.ply"
        
    Example:
        >>> points = np.random.rand(1000, 3)
        >>> save_points(points, "output/cloud.ply")
    """
    import open3d as o3d
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, point_cloud)

def save_points_to_ply(points, filename):
    """
    Save PyTorch point tensor to PLY file.
    
    Converts PyTorch tensor to Open3D point cloud and saves in PLY format.
    PLY is a standard 3D file format readable by most 3D software.
    
    Args:
        points (torch.Tensor): Points to save, shape (N, 3)
        filename (str): Output file path (should end with .ply)
    
    Raises:
        ValueError: If points not shape (N, 3)
        
    Example:
        >>> points = torch.rand(1000, 3)
        >>> save_points_to_ply(points, "output.ply")
    """
    # Validate input shape
    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError("Input tensor must be of shape (N, 3)")

    # Convert to NumPy
    points_np = points.detach().cpu().numpy()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Save to file
    o3d.io.write_point_cloud(filename, pcd)


def save_mesh_as_ply(mesh, file_path):
    """
    Save PyTorch3D mesh to PLY file via Open3D.
    
    Extracts vertices, faces, and colors from PyTorch3D mesh format
    and saves as PLY using Open3D.
    
    Args:
        mesh: PyTorch3D Meshes object
        file_path (str): Output file path
        
    Example:
        >>> from pytorch3d.structures import Meshes
        >>> # ... create mesh ...
        >>> save_mesh_as_ply(mesh, "output.ply")
    """
    # Extract mesh components
    vertices = mesh.verts_padded()[0].cpu().numpy()
    faces = mesh.faces_padded()[0].cpu().numpy()
    colors = mesh.textures.verts_features_packed().cpu().numpy()

    # Create Open3D mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Save to file
    o3d.io.write_triangle_mesh(file_path, o3d_mesh)
    print(f"Mesh saved to {file_path}")

class IOUtils:
    """Utilities for input/output operations"""
    
    @staticmethod
    def load_functionalities(handler):
        """Load object functionalities from file"""
        if os.path.exists(os.path.join(handler.save_in, 'functionalities.pt')):
            functionalities = torch.load(
                os.path.join(handler.save_in, 'functionalities.pt')
            )
            for obj_id, obj in enumerate(handler.objects):
                obj.functionality = functionalities[obj_id]
                
    @staticmethod
    def update_functionalities(handler):
        """Update and save object functionalities"""
        if os.path.exists(os.path.join(handler.save_in, 'functionalities.pt')):
            functionalities = torch.load(
                os.path.join(handler.save_in, 'functionalities.pt')
            )
            for obj_id, obj in enumerate(handler.objects):
                if (functionalities[obj_id] == obj.obj_name and
                    obj.functionality != obj.obj_name):
                    functionalities[obj_id] = obj.functionality
            torch.save(functionalities, os.path.join(handler.save_in, 'functionalities.pt'))
        else:
            torch.save(
                [obj.functionality for obj in handler.objects],
                os.path.join(handler.save_in, 'functionalities.pt')
            )
            
    @staticmethod
    def log_pose(handler, T, groups, epoch, append_suffix_to_path='_first_stage'):
        """Log object poses to file"""
        save_in = handler.save_in_out_steps / ('poses' + append_suffix_to_path)
        poses = []
        for i, (transformed_group, t) in enumerate(zip(groups, T)):
            for g_ele_i in transformed_group:
                if (g_ele_i == transformed_group[0]) and not handler.objects[g_ele_i].train:
                    continue
                poses.append((g_ele_i, t))
        os.makedirs(save_in, exist_ok=True)
        torch.save(poses, save_in / (str(epoch) + '.pt'))
