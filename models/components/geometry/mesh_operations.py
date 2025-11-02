import numpy as np
import open3d as o3d
import copy


class MeshOperations:
    """Operations for manipulating mesh data"""
    
    @staticmethod
    def update_point_cloud(data, m_id, T, m_center, class_agnostic_masks):
        """
        Update point cloud with transformation
        
        Args:
            data: O3D mesh data
            m_id: Mask ID for object to transform
            T: Transformation matrix
            m_center: Mask ID for transformation center
            class_agnostic_masks: Array of object masks
        """
        MeshOperations.update_vertices(
            data, class_agnostic_masks[m_id], T, class_agnostic_masks[m_center]
        )
        MeshOperations.update_normals(data, class_agnostic_masks[m_id], T)
        MeshOperations.update_faces(data, class_agnostic_masks[m_id])
        
    @staticmethod
    def update_faces(data, mask):
        """Remove faces connected to masked vertices"""
        faces = np.asarray(data.triangles)
        point_indx = np.arange(len(np.asarray(data.vertices)))
        inst_indices = point_indx[mask]
        incl_mask = np.isin(faces, inst_indices).sum(axis=-1)
        incl_mask = ((incl_mask == 1) + (incl_mask == 2)) >= 1
        faces = faces[~incl_mask]
        data.triangles = o3d.utility.Vector3iVector(faces)
        
    @staticmethod
    def update_vertices(data, mask, T, mask_center):
        """Update vertex positions with transformation"""
        points = np.asarray(data.vertices)
        masked_points = points[mask]
        ones = np.ones((masked_points.shape[0], 1))
        masked_points_homogeneous = np.hstack((masked_points, ones))
        
        center = np.hstack((points[mask_center], np.ones((mask_center.sum(), 1)))).mean(axis=0)
        center[-1] = 0
        
        transformed_points_homogeneous = (T @ (masked_points_homogeneous - center).T).T + center
        transformed_points = transformed_points_homogeneous[:, :3]
        
        points[mask] = transformed_points
        data.vertices = o3d.utility.Vector3dVector(points)
        
    @staticmethod
    def update_normals(data, mask, T):
        """Update normal vectors with rotation from transformation"""
        normals = np.asarray(data.vertex_normals)
        rotation_matrix = T[:3, :3]
        masked_normals = normals[mask]
        transformed_normals = (rotation_matrix @ masked_normals.T).T
        transformed_normals = transformed_normals / np.linalg.norm(
            transformed_normals, axis=1, keepdims=True
        )
        normals[mask] = transformed_normals
        data.vertex_normals = o3d.utility.Vector3dVector(normals)
        
    @staticmethod
    def remove_objects(data, remove_masks, scene_center):
        """
        Remove objects from mesh by masking
        
        Args:
            data: O3D mesh data
            remove_masks: List of masks to remove
            scene_center: Center of scene
        """
        mask = sum([remove_masks[m_ir] for m_ir in range(len(remove_masks))]) >= 1
        points = np.asarray(data.vertices)
        points[mask] = scene_center
        faces = np.asarray(data.triangles)
        point_indx = np.arange(len(np.asarray(data.vertices)))
        inst_indices = point_indx[mask]
        incl_mask = np.isin(faces, inst_indices).sum(axis=-1) > 0
        faces = faces[~incl_mask]
        data.triangles = o3d.utility.Vector3iVector(faces)
        data.vertices = o3d.utility.Vector3dVector(points)


class DummyDataManager:
    """Manages dummy data copies for visualization"""
    
    def __init__(self, handler):
        self.handler = handler
        self.data_dummy = None
        
    def initialize_dummy_data(self, num_dummies):
        """Create dummy data copies"""
        self.data_dummy = [copy.deepcopy(self.handler.data) for _ in range(num_dummies)]
        self.handler.data_dummy = self.data_dummy
        
    def remove_dummy_data(self):
        """Remove dummy data"""
        self.data_dummy = None
        self.handler.data_dummy = None
        
    def update_dummy_point_cloud(self, dummy_index, m_id, T, m_center):
        """Update specific dummy data"""
        MeshOperations.update_point_cloud(
            self.data_dummy[dummy_index],
            m_id, T, m_center,
            self.handler.class_agnostic_masks
        )

def create_spheres_with_normals(points, radius=0.1):
    """
    Create sphere meshes at each point for visualization.
    
    Useful for visualizing point clouds as solid spheres with proper
    lighting/shading. Each sphere has computed vertex normals for rendering.
    
    Args:
        points (np.ndarray): Point positions, shape (N, 3)
        radius (float): Sphere radius in meters. Default: 0.1
    
    Returns:
        list of o3d.geometry.TriangleMesh: Sphere meshes (red colored)
        
    Example:
        >>> points = np.array([[0,0,0], [1,0,0], [0,1,0]])
        >>> spheres = create_spheres_with_normals(points, radius=0.05)
        >>> o3d.visualization.draw_geometries(spheres)
    """
    spheres = []
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1, 0, 0])  # Red
        spheres.append(sphere)
    return spheres


def create_sphere_at_point(center, radius=0.1, resolution=20):
    """
    Create single sphere mesh at specific location.
    
    Args:
        center (tuple or np.ndarray): Sphere center (x, y, z)
        radius (float): Sphere radius. Default: 0.1
        resolution (int): Mesh resolution (higher = smoother). Default: 20
    
    Returns:
        o3d.geometry.TriangleMesh: Sphere mesh with computed normals
        
    Example:
        >>> sphere = create_sphere_at_point((1.5, 2.0, 0.5), radius=0.05)
        >>> o3d.visualization.draw_geometries([sphere])
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, 
        resolution=resolution
    )
    sphere.translate(center)
    sphere.compute_vertex_normals()
    return sphere


def return_points_as_mesh_spheres(points, radius=0.1):
    """
    Convert point array to list of sphere meshes.
    
    Similar to create_spheres_with_normals but with configurable resolution.
    
    Args:
        points (np.ndarray): Point positions, shape (N, 3)
        radius (float): Sphere radius. Default: 0.1
    
    Returns:
        list of o3d.geometry.TriangleMesh: Sphere meshes
        
    Example:
        >>> points = np.random.rand(50, 3)
        >>> spheres = return_points_as_mesh_spheres(points, radius=0.02)
    """
    spheres = []
    for point in points:
        sphere = create_sphere_at_point(point, radius)
        spheres.append(sphere)
    return spheres


def get_complementary_mesh(data, mask):
    """
    Extract mesh subset using boolean vertex mask.
    
    Creates new mesh containing only vertices where mask is True,
    along with faces that have all vertices in the mask. Useful for
    isolating specific parts of a segmented mesh.
    
    Args:
        data (o3d.geometry.TriangleMesh): Input mesh
        mask (np.ndarray): Boolean mask for vertices, shape (N,)
    
    Returns:
        o3d.geometry.TriangleMesh: Filtered mesh
        
    Example:
        >>> # Keep only top half of mesh
        >>> vertices = np.asarray(mesh.vertices)
        >>> mask = vertices[:, 2] > vertices[:, 2].mean()
        >>> top_mesh = get_complementary_mesh(mesh, mask)
    
    Note:
        Vertices outside mask are set to origin (0,0,0) but still present.
        Only faces with ALL vertices in mask are kept.
    """
    data_new = o3d.geometry.TriangleMesh()
    
    # Deep copy mesh attributes
    vertices = copy.deepcopy(np.asarray(data.vertices))
    triangles = copy.deepcopy(np.asarray(data.triangles))
    vertex_colors = copy.deepcopy(np.asarray(data.vertex_colors))
    vertex_normals = copy.deepcopy(np.asarray(data.vertex_normals))

    # Zero out masked vertices (outside selection)
    vertices[~mask] = 0
    vertex_colors[~mask] = 0
    vertex_normals[~mask] = 0

    # Keep only faces where all 3 vertices are in mask
    mask_triangles = np.all(mask[triangles], axis=1)
    triangles_filtered = triangles[mask_triangles]

    # Assign to new mesh
    data_new.vertices = o3d.utility.Vector3dVector(vertices)
    data_new.triangles = o3d.utility.Vector3iVector(triangles_filtered)
    data_new.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    data_new.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    
    return data_new


def oversegment_cc_and_plot(mesh):
    """
    Segment mesh into connected components and return vertex masks.
    
    Uses triangle connectivity to identify separate mesh parts (like detecting
    individual objects in a combined mesh). Assigns random colors for visualization.
    
    Algorithm:
        1. Compute connected components based on triangle adjacency
        2. Create binary mask for each component's vertices
        3. Filter out small components (<500 vertices)
        4. Assign random colors to remaining components
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Input mesh to segment
    
    Returns:
        np.ndarray: Boolean masks, shape (num_components, num_vertices)
                   Each row is a mask for one component
        
    Example:
        >>> mesh = o3d.io.read_triangle_mesh("scene.ply")
        >>> masks = oversegment_cc_and_plot(mesh)
        >>> print(f"Found {len(masks)} components")
        >>> # Visualize: mesh now has colored components
        >>> o3d.visualization.draw_geometries([mesh])
    
    Side effects:
        Modifies mesh.vertex_colors to show segmentation
    """
    # Compute connected components
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles()
    )

    # Get cluster ID for each triangle
    triangle_labels = np.asarray(triangle_clusters)
    max_label = triangle_labels.max()

    print(f"Mesh has {max_label + 1} connected components.")

    # Initialize vertex masks (one per component)
    num_vertices = len(mesh.vertices)
    masks = [np.zeros(num_vertices, dtype=bool) for _ in range(max_label + 1)]

    # Assign vertices to components based on their triangles
    for triangle_id, cluster_id in enumerate(triangle_labels):
        for vertex_id in mesh.triangles[triangle_id]:
            masks[cluster_id][vertex_id] = True

    # Filter out small components (< 500 vertices)
    masks = np.stack(masks)
    masks = masks[masks.sum(axis=1) > 500]

    # Assign random colors for visualization
    colors = np.random.rand(len(masks) + 1, 3)
    vertex_colors = np.zeros((num_vertices, 3))
    
    for cluster_id, mask in enumerate(masks):
        vertex_colors[mask] = colors[cluster_id]
    
    # Apply colors to mesh
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    return masks