"""
Loss Functions for 3D Scene Optimization

This module contains various loss functions used during scene optimization to ensure
objects are placed correctly according to spatial constraints. The losses guide the
optimization process to find valid object placements.

Mathematical Background:
- Most losses use a hinge loss formulation: max(0, violation)
- This means: no penalty when constraint satisfied, linear penalty when violated
- Signed distances are used for polygon containment (positive = outside, negative = inside)
- Cosine similarity measures alignment between vectors (1 = aligned, -1 = opposite)

Key Concepts:
1. Spatial Constraints: Objects must be "on top of", "inside", "against" surfaces
2. Collision Avoidance: Objects shouldn't overlap with scene or other objects
3. Alignment: Objects should face correct directions
4. Distance Preservation: Related objects should maintain reasonable distances
"""

import torch
import numpy as np
import fpsample
from models.components.visualization.debug_viz import plot_polygon_and_points, plot_contour_with_normals

def max_point_resolution(point_cloud_1, point_cloud_2):
    # Compute pairwise distances
    dists_1 = torch.cdist(point_cloud_1, point_cloud_1)
    dists_2 = torch.cdist(point_cloud_2, point_cloud_2)
    
    # Find the minimum distance for each point to its nearest neighbor
    # Set diagonal to infinity to ignore self-distances
    dists_1.fill_diagonal_(float('inf'))
    dists_2.fill_diagonal_(float('inf'))
    
    min_dists_1, _ = dists_1.min(dim=1)
    min_dists_2, _ = dists_2.min(dim=1)
    
    # Find the maximum resolution from both point clouds
    max_resolution = max(min_dists_1.max().item(), min_dists_2.max().item())
    
    return max_resolution
def get_distances(source_points, target_points, max_points=200, fps=False, p_norm=2):
    """
    Compute distances between sampled source and target point clouds.
    
    Performs intelligent sampling and filtering to focus on relevant collision regions:
    1. Sample points (random or farthest point sampling)
    2. Filter target points to source bounding box
    3. Filter by Z-height overlap between clouds
    4. Compute pairwise distances
    
    Args:
        source_points (torch.Tensor): Source point cloud, shape (N, 3)
        target_points (torch.Tensor): Target point cloud, shape (M, 3)
        max_points (int): Maximum points to sample from each cloud. Default: 200
        fps (bool): If True, use farthest point sampling; else random sampling
        p_norm (int): Norm for distance computation (1=Manhattan, 2=Euclidean)
    
    Returns:
        Tuple of:
            - distances (torch.Tensor): Pairwise distance matrix, shape (n, m)
            - ref_distance (float): Reference distance for normalization
            - no_collision (bool): True if no points in collision zone
            - sampled_source (torch.Tensor): Filtered source points used
            - sampled_target (torch.Tensor): Filtered target points used
    """
    # Sample points from both clouds
    if fps:
        # Farthest point sampling for better coverage
        source_sampled_indices = fpsample.fps_sampling(source_points, max_points)
        target_sampled_indices = fpsample.fps_sampling(target_points, max_points)
    else:
        # Random sampling
        num_source = min(max_points, len(source_points))
        num_target = min(max_points, len(target_points))
        source_sampled_indices = torch.randperm(len(source_points))[:num_source]
        target_sampled_indices = torch.randperm(len(target_points))[:num_target]
    
    # Calculate reference distance (characteristic scale of point clouds)
    ref_distance = max_point_resolution(
        source_points[source_sampled_indices].detach(),
        target_points[target_sampled_indices].detach()
    )
    
    # Get sampled points
    filtered_source_points = source_points[source_sampled_indices]
    filtered_target_points = target_points[target_sampled_indices]
    
    # Filter target points to source bounding box (XY only)
    s_min_p = filtered_source_points.min(dim=0).values
    s_max_p = filtered_source_points.max(dim=0).values
    target_in_bbox = (
        (filtered_target_points[:, 0] > s_min_p[0]) &
        (filtered_target_points[:, 0] < s_max_p[0]) &
        (filtered_target_points[:, 1] > s_min_p[1]) &
        (filtered_target_points[:, 1] < s_max_p[1])
    )
    filtered_target_points = filtered_target_points[target_in_bbox]
    
    # Define Z-height bounds for filtering (use full height range: p=0.0)
    p = 0.0
    min_bound_s = source_points[:, -1].min() + (source_points[:, -1].max() - source_points[:, -1].min()) * p
    max_bound_s = source_points[:, -1].max() - (source_points[:, -1].max() - source_points[:, -1].min()) * p
    min_bound_t = target_points[:, -1].min() + (target_points[:, -1].max() - target_points[:, -1].min()) * p
    max_bound_t = target_points[:, -1].max() - (target_points[:, -1].max() - target_points[:, -1].min()) * p
    
    # Filter source points by target Z-bounds (keep source points at target's height)
    source_z_mask = (
        (filtered_source_points[:, 2] > min_bound_t) &
        (filtered_source_points[:, 2] < max_bound_t)
    )
    inbound_s_pt = filtered_source_points[source_z_mask]
    
    # Check if any source points remain after filtering
    if len(inbound_s_pt) == 0:
        return 0.0, 0.0, True, None, None
    
    # Filter target points by source Z-bounds (keep target points at source's height)
    target_z_mask = (
        (filtered_target_points[:, 2] > min_bound_s) &
        (filtered_target_points[:, 2] < max_bound_s)
    )
    inbound_t_pt = filtered_target_points[target_z_mask]
    
    # Quick collision check: compute distances with Z filtering
    distances_quick = torch.cdist(inbound_s_pt, inbound_t_pt, p=2)
    collision_detected = (distances_quick < 4 * ref_distance).any()
    
    if not collision_detected:
        return distances_quick, ref_distance, True, None, None
    
    # If collision detected, use full source points (remove Z filter on source)
    inbound_s_pt = filtered_source_points
    
    # Compute final pairwise distances
    distances = torch.cdist(inbound_s_pt, inbound_t_pt, p=p_norm)
    
    return distances, ref_distance, False, inbound_s_pt, inbound_t_pt

def signed_distance(polygon_points, polygon_normals, points, k=3, plot_progress=False, return_mean=True):
    """
    Calculate signed distance field from points to a 2D polygon boundary.
    
    Computes how far each point is from the polygon boundary, with sign indicating
    inside (negative) or outside (positive). Uses k-nearest polygon vertices for
    robustness to irregular shapes.
    
    Args:
        polygon_points (torch.Tensor): Polygon vertices, shape (N, 2) or (N, 3).
                                      Only XY coordinates are used.
        polygon_normals (torch.Tensor): Inward-pointing normals at vertices, shape (N, 2) or (N, 3).
        points (torch.Tensor): Query points to test, shape (M, 2) or (M, 3).
        k (int): Number of nearest polygon vertices to average. Default: 3.
        plot_progress (bool): If True, visualize polygon and points for debugging.
        return_mean (bool): If True, return mean distance over k neighbors (M,).
                           If False, return all k distances (M, k).
    
    Returns:
        torch.Tensor: Signed distances. Negative = inside, positive = outside.
    """
    # Move polygon data to same device as query points
    polygon_points = polygon_points.to(points.device)
    polygon_normals = polygon_normals.to(points.device)
    
    # Reshape for broadcasting: polygon (1, N, dim), points (M, 1, dim)
    polygon_points = polygon_points.unsqueeze(0)   # (1, N, 2 or 3)
    polygon_normals = polygon_normals.unsqueeze(0) # (1, N, 2 or 3)
    points = points.unsqueeze(1)                   # (M, 1, 2 or 3)
    
    # Compute vector from each polygon vertex to each point (XY only)
    vector_diff = points - polygon_points[..., :2]  # (M, N, 2)
    
    # Find k nearest polygon vertices for each point
    distances_to_vertices = torch.norm(vector_diff, dim=-1)  # (M, N)
    _, top_k_indices = torch.topk(distances_to_vertices, k, dim=1, largest=False)  # (M, k)
    
    # Extract vectors and normals for k nearest vertices
    expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, 2)  # (M, k, 2)
    vector_diff_topk = torch.gather(vector_diff, 1, expanded_indices)  # (M, k, 2)
    normals_topk = torch.gather(
        polygon_normals.expand(points.size(0), -1, -1), 1, expanded_indices
    )  # (M, k, 2)
    
    # Project vector onto normal: signed_dist = (v · n) / ||n||
    signed_distances = torch.sum(vector_diff_topk * normals_topk, dim=2) / torch.norm(normals_topk, dim=2)  # (M, k)
    
    # Optional debug visualization
    if plot_progress:
        plot_polygon_and_points(
            polygon_points.squeeze()[..., :2].detach().cpu(),
            points.squeeze().detach().cpu()
        )
        plot_contour_with_normals(
            polygon_points.squeeze()[..., :2].detach().cpu(),
            polygon_normals.squeeze()[..., :2].detach().cpu()
        )
    
    # Return mean or all k distances
    if return_mean:
        return signed_distances.mean(dim=1)  # (M,)
    else:
        return signed_distances  # (M, k)

def on_top_of_loss_bbox(points_source, points_target):
    loss = torch.tensor(0.0).to(points_source.device)
    loss += torch.norm(points_source.mean(dim=0)[:2]-points_target.mean(dim=0)[:2].to(points_source.device))
    loss += torch.norm(points_source[:, -1].min(dim=0).values-points_target[:, -1].max(dim=0).values.to(points_source.device))
    return loss
def normal_alignment_loss(updated_normal, target_normal):
    """
    Measure misalignment between two normal vectors.
    
    Math:
        loss = |dot(n1, n2) + 1|
        
    When vectors are opposite (facing each other), dot product = -1, so loss = 0.
    When vectors are aligned (same direction), dot product = 1, so loss = 2.
    
    Use case: Ensures object faces toward a target (e.g., chair facing table)
    
    Args:
        updated_normal: Current normal vector (3D)
        target_normal: Desired normal vector (3D)
        
    Returns:
        Scalar loss (0 when vectors are opposite)
    """
    # Normalize to unit vectors
    updated_normal = updated_normal / torch.norm(updated_normal)
    target_normal = target_normal / torch.norm(target_normal)

    # Dot product measures alignment
    dot_product = updated_normal.T @ target_normal
    
    # We want them opposite (dot = -1), so penalize deviation from -1
    loss = torch.abs(dot_product + 1)
    return loss


def pull_loss(updated_center, target_center, margin=1.0):
    """
    Encourage objects to be close to targets (but not too close).
    
    Math:
        distance = ||center1 - center2||
        loss = distance (only applied if distance > 0.2m)
        
    The margin parameter isn't used in implementation but represents the
    desired proximity threshold.
    
    Use case: Pull objects closer during "close to" constraint
    
    Args:
        updated_center: Current object center (3D)
        target_center: Target centers (N x 3)
        margin: Desired minimum distance (unused in current implementation)
        
    Returns:
        Mean distance loss (0 when within 0.2m)
    """
    distance = torch.norm(updated_center[None, ...] - target_center[:, None, :], dim=-1)
    
    # Only apply loss if mean distance > 0.2m (objects are far)
    return distance.mean() * (distance.detach().mean() > 0.2)


def facing_loss(updated_normal, target_normal, source_points, target_points, margin=1.0):
    """
    Ensure object faces toward target object.
    
    Math:
        1. Find closest point pair between objects
        2. Compute direction vector from source to target
        3. Measure alignment: loss = 1 - cos(normal, direction)
        
    Use case: Chair facing table, TV facing couch
    
    Args:
        updated_normal: Object's current front normal
        target_normal: Target's front normal (unused currently)
        source_points: Points of object being placed
        target_points: Points of target object
        margin: Unused parameter
        
    Returns:
        Cosine loss weighted by 20x for emphasis
    """
    # Find closest points between objects
    dists = torch.cdist(source_points, target_points)
    min_index = torch.argmin(dists)
    min_source_id, min_target_id = divmod(min_index.item(), dists.size(1))
    
    # Direction from source to target
    dir_vec = target_points[min_target_id] - source_points[min_source_id]
    
    loss = torch.tensor(0.0, device=updated_normal.device, dtype=torch.float64)
    loss += 20 * cosine_loss(updated_normal, dir_vec.detach())
    
    return loss


def inside_polygon_loss(polygon_points, polygon_normals, points, return_mean=True):
    """
    Penalize points outside a 2D polygon boundary.
    
    Math (Signed Distance):
        For each point p and polygon edge (v_i, v_{i+1}) with normal n:
        signed_dist = dot(p - v_i, n)
        - If signed_dist < 0: point is inside (safe)
        - If signed_dist > 0: point is outside (violation)
        
        loss = sum(max(0, signed_dist)) / num_violations
        
    This uses a hinge loss: only points outside contribute to loss.
    The signed distance measures how far outside the boundary.
    
    Use case: Ensure object stays within table surface, shelf bounds, floor area
    
    Args:
        polygon_points: Boundary vertices (N x 2 or N x 3)
        polygon_normals: Inward-pointing normals at each vertex (N x 2 or N x 3)
        points: Points to check (M x 2 or M x 3)
        return_mean: If True, return mean loss; else return per-point losses
        
    Returns:
        Scalar mean loss or vector of per-point losses
    """
    # Calculate signed distances (positive = outside, negative = inside)
    distances = signed_distance(polygon_points, polygon_normals, points)

    # Hinge loss: only penalize points outside (positive distances)
    loss = torch.clamp(distances, min=0)
    
    if return_mean:
        # Average over violating points (avoid division by zero)
        return loss.sum() / (loss != 0).sum() if (loss != 0).sum() != 0 else loss.sum()
    else:
        return loss


def regression_loss(x, y):
    """
    Simple L2 distance between vectors.
    
    Math:
        loss = ||x - y||_2
        
    Use case: Match elevations, align positions
    """
    return torch.norm(x - y)


def close_to_loss(points_source, points_target, num_samples=2000, margin=0.4):
    """
    Ensure minimum distance between object point clouds.
    
    Math:
        1. Sample points from each cloud
        2. Find minimum pairwise distance
        3. loss = max(0, min_distance - margin)
        
    Hinge loss ensures objects stay at least 'margin' meters apart.
    
    Use case: "Place chair close to table" constraint
    
    Args:
        points_source: Source object points (N x 3)
        points_target: Target object points (M x 3)
        num_samples: Number of points to sample for efficiency
        margin: Desired minimum distance (0.4m)
        
    Returns:
        Scalar loss (0 when objects are within margin distance)
    """
    # Sample points for efficiency
    indices_source = torch.randperm(points_source.shape[0])[:num_samples]
    sampled_points_source = points_source[indices_source]

    indices_target = torch.randperm(points_target.shape[0])[:num_samples]
    sampled_points_target = points_target[indices_target]

    # Find minimum distance between sampled points
    min_dist = torch.cdist(sampled_points_source, sampled_points_target).min()
    
    # Penalize if farther than margin
    return torch.clamp(min_dist - margin, min=0)


def far_from_loss(cs, ct, m):
    """
    Ensure objects maintain minimum separation.
    
    Math:
        loss = max(0, m - ||cs - ct||)
        
    Inverse of close_to_loss: penalizes being too close.
    
    Args:
        cs: Source center
        ct: Target center
        m: Minimum allowed distance
        
    Returns:
        Scalar loss (0 when objects are at least m apart)
    """
    return torch.clamp(m - torch.norm(cs - ct), min=0)


def on_top_of_loss(polygon_points, polygon_normals, surface_elevation, 
                   object_min_point, points, in_p_fac=1., reg_fac=0.1, 
                   return_seperate=False):
    """
    Constrain object to be on top of a surface.
    
    Math:
        Two components:
        1. Boundary constraint: inside_polygon_loss ensures XY containment
        2. Elevation constraint: ||surface_z - object_base_z|| (currently disabled)
        
        loss = in_p_fac * boundary_loss + reg_fac * elevation_loss
        
    The elevation loss is currently disabled (commented out) but would ensure
    the object sits exactly at the surface height.
    
    Use case: "Place book on table", "Put cup on shelf"
    
    Args:
        polygon_points: Surface boundary in XY (N x 2)
        polygon_normals: Inward normals (N x 2)
        surface_elevation: Height of surface (scalar)
        object_min_point: Lowest Z coordinate of object (scalar)
        points: Object's base points in XY (M x 2)
        in_p_fac: Weight for boundary loss (default 1.0)
        reg_fac: Weight for elevation loss (default 0.1, currently unused)
        return_seperate: Return components separately if True
        
    Returns:
        Combined scalar loss or tuple of (boundary_loss, elevation_loss)
    """
    if return_seperate:
        return (inside_polygon_loss(polygon_points, polygon_normals, points),
                regression_loss(surface_elevation.to(points.device), object_min_point))
    
    loss = torch.tensor([0.0]).to(points.device)
    
    # Ensure object base is within surface boundary
    loss += in_p_fac * inside_polygon_loss(polygon_points, polygon_normals, points)
    
    # Elevation matching (currently disabled for stability)
    # Uncomment to enforce exact height matching:
    # elevation_loss = regression_loss(surface_elevation.to(points.device), object_min_point)
    # loss += reg_fac * elevation_loss * (elevation_loss.detach() > 0.005)
    
    return loss


def in_loss(surfaces, object_min_point, anchor_points, in_p_fac=1.0, 
            reg_fac=0.003, t=None, center=None):
    """
    Multi-surface version of on_top_of_loss.
    
    Used when object could be placed on any of several surfaces (e.g., 
    different shelves in a bookcase). Computes loss for each surface and
    returns the average.
    
    Math:
        loss = (1/N) * sum_i on_top_of_loss(surface_i, ...)
        
    If transformation t is provided, surfaces are transformed first.
    
    Args:
        surfaces: List of surface dictionaries with 'contour points', 
                 'contour normals', 'elevation'
        object_min_point: Base height of object
        anchor_points: Object's base points
        in_p_fac: Boundary loss weight
        reg_fac: Elevation loss weight
        t: Optional transformation network
        center: Center for transformation
        
    Returns:
        Average loss across all surfaces
    """
    loss = torch.tensor([0.0]).to(anchor_points.device)
    
    for s in surfaces:
        if t is not None:
            # Transform surface with learned transformation
            transformated_contour = t.forward_xy(
                s['contour points'], center=center
            ).detach()[:, :2]
            transformated_normals = t.forward_xy_rotate(
                s['contour normals']
            ).detach()[:, :2]
            transformated_elevation = t.forward_z(
                s['elevation']
            ).detach().squeeze()[-1]
            
            loss += on_top_of_loss(
                transformated_contour, transformated_normals, 
                transformated_elevation, object_min_point, 
                anchor_points, in_p_fac, reg_fac
            )
        else:
            # Use surface as-is
            loss += on_top_of_loss(
                s['contour points'], s['contour normals'], 
                s['elevation'][-1], object_min_point, 
                anchor_points, in_p_fac, reg_fac
            )
    
    return loss / len(surfaces)


def cosine_loss(vec, vec_target):
    """
    Measure misalignment between 2D vectors using cosine similarity.
    
    Math:
        cos(θ) = dot(v1, v2) / (||v1|| * ||v2||)
        loss = 1 - cos(θ)
        
    When vectors are aligned: cos(θ) = 1, loss = 0
    When vectors are perpendicular: cos(θ) = 0, loss = 1
    When vectors are opposite: cos(θ) = -1, loss = 2
    
    Only uses XY components (ignores Z), suitable for orientation matching.
    
    Use case: Align object direction with target direction
    
    Args:
        vec: Current vector (3D, uses only [:2])
        vec_target: Target vector (3D, uses only [:2])
        
    Returns:
        Scalar loss in [0, 2]
    """
    # Normalize vectors
    vec_norm = vec / vec.norm(dim=0, keepdim=True)
    vec_target_norm = vec_target / vec_target.norm(dim=0, keepdim=True)
    
    # Compute cosine similarity (XY plane only)
    cosine_similarity = torch.dot(vec_norm[:2], vec_target_norm[:2].to(vec_norm.device))
    
    # Loss is 1 - similarity (0 when aligned)
    loss = 1 - cosine_similarity
    
    return loss


def minimum_distance_loss(pc1, pc2):
    """
    Sum of minimum distances from each point in pc1 to pc2.
    
    Math:
        For each point p in pc1:
            d(p) = min_{q in pc2} ||p - q||
        loss = sum_p d(p)
        
    Only uses XY coordinates, ignoring height differences.
    
    Use case: Encourage point clouds to be close in XY plane
    
    Args:
        pc1: First point cloud (N x 3)
        pc2: Second point cloud (M x 3)
        
    Returns:
        Sum of minimum distances
    """
    # Calculate pairwise distances in XY
    distances = torch.cdist(pc1[:, :2], pc2[:, :2].to(pc1.device))

    # Get minimum distance for each point in pc1
    min_distances = distances.min(dim=-1).values.sum()

    return min_distances


def minimum_signed_distance_loss(pc1, pc2, normal):
    """
    Signed distance loss for collision avoidance.
    
    Math:
        1. For each point p in pc1, find nearest point q in pc2
        2. Compute direction vector: dir = p - q
        3. Signed distance: sd = dot(dir, normal)
           - If sd > 0: p is on positive side of surface (safe)
           - If sd < 0: p is penetrating surface (collision)
        4. loss = sum(max(0, -sd)) / num_violations
        
    The normal defines "inside" vs "outside". We penalize negative
    signed distances (penetration).
    
    Use case: Prevent objects from intersecting walls/surfaces
    
    Args:
        pc1: Points to check (N x 3)
        pc2: Surface points (M x 3)
        normal: Surface normal defining inside/outside (3D)
        
    Returns:
        Tuple of (loss, (collision_point_source, collision_point_target))
    """
    # Find direction vectors to nearest neighbors
    dir_vec = pc1[:, None, :3] - pc2[None, ::10, :3].to(pc1.device)
    
    # Find closest points
    _, top_k_signed_distances_indx = torch.topk(
        torch.norm(dir_vec, dim=-1), 1, dim=1, largest=False
    )
    
    # Get direction vectors for closest pairs
    vector_diff_topk = torch.gather(
        dir_vec, 1, 
        top_k_signed_distances_indx.unsqueeze(-1).expand(-1, -1, 3)
    )
    
    # Compute signed distances
    signed_dist = (vector_diff_topk * normal[None, None, :].to(pc1.device)).sum(dim=-1)
    
    # Penalize negative signed distances (penetration)
    loss = torch.clamp(-signed_dist, min=0)
    loss = loss.sum() / (loss != 0).sum() if (loss != 0).sum() != 0 else loss[0, 0]
    
    # Find worst collision point for visualization
    pt_id = torch.argmin(signed_dist.squeeze())

    return (loss, 
            (pc1[pt_id].detach().cpu().numpy(), 
             pc2[::10][top_k_signed_distances_indx[pt_id.detach().cpu()].detach().cpu()].numpy().squeeze()))


def against_loss(dominant_normal, corners, dominant_normal_target, 
                points_target, plot_normals_flag=False):
    """
    Constrain object to be "against" a surface (e.g., painting on wall).
    
    Math:
        Two components:
        1. Alignment: 15 * cosine_loss(normal, target_normal)
           Ensures object faces same direction as surface
        2. Contact: minimum_signed_distance_loss(corners, surface)
           Ensures object touches surface without penetrating
           
        loss = 15 * alignment_loss + contact_loss
        
    High weight (15x) on alignment ensures strong orientation constraint.
    
    Use case: "Hang painting on wall", "Push desk against wall"
    
    Args:
        dominant_normal: Object's front normal
        corners: Object's corner points
        dominant_normal_target: Surface normal
        points_target: Surface points
        plot_normals_flag: Debug visualization flag
        
    Returns:
        Tuple of (total_loss, worst_collision_points)
    """
    loss = torch.tensor([0.0]).to(dominant_normal.device)
    
    # Strong penalty for misalignment
    coss_loss = cosine_loss(dominant_normal, dominant_normal_target)
    loss += 15 * coss_loss
    
    # Ensure contact without penetration
    reg_loss, max_loss_points = minimum_signed_distance_loss(
        corners, points_target, dominant_normal_target
    )
    loss += reg_loss
    
    if plot_normals_flag:
        from models.components.visualization.debug_viz import plot_normals
        plot_normals(dominant_normal[:2], dominant_normal_target[:2], name='alignment')
    
    return loss, max_loss_points



def batched_tracking_loss(points_source, points_target, batch_size=500):
    """
    Compute minimum distance between point clouds in batches.
    
    Used for memory efficiency when point clouds are large.
    
    Math:
        For each batch of source points:
            Find minimum distance to any target point
        Return global minimum
        
    Args:
        points_source: Source points (N x 3)
        points_target: Target points (M x 3)
        batch_size: Points per batch
        
    Returns:
        Minimum distance across all point pairs
    """
    min_distances = []
    for i in range(0, points_source.shape[0], batch_size):
        batch = points_source[i:i + batch_size]
        dists = torch.cdist(batch, points_target)
        min_distances.append(dists.min().cpu())
    
    return torch.cat(min_distances).min()


def collision_loss(points_source, points_target, targets_centers, num_points=8, 
                   floor_cont_points_normals=None):
    """
    Main collision avoidance loss for scene optimization.
    
    Algorithm:
        1. Create expanded bounding box around source (2x size)
        2. Filter target points inside this box
        3. Sample points from both clouds
        4. Check if minimum distance < threshold (collision detected)
        5. If collision: penalize distance + center-to-center distances
        
    Math:
        If tracking_loss (collision detected):
            point_loss = max(0, 50 - min_distance)
            center_loss = weighted_sum(max(0, 50 - center_distances))
            loss = point_loss + center_loss
        Else:
            loss = 0
            
    The "50" threshold creates a soft margin - objects start being penalized
    when they get within 50 units (effectively creating a collision buffer).
    
    Args:
        points_source: Object being placed (N x 3)
        points_target: Scene/other objects (M x 3)
        targets_centers: Centers of other movable objects (K x 3)
        num_points: Number of points to sample
        floor_cont_points_normals: Optional floor boundary (points, normals)
        
    Returns:
        Tuple of:
            - loss: Scalar collision loss
            - tracking_loss: Boolean, True if collision detected
            - anchor_point_source: Source collision point (for viz)
            - anchor_point_target: Target collision point (for viz)
            - num_target_points: Number of points in collision region
    """
    # Create expanded bounding box (2x size)
    max_source = points_source.max(dim=0).values
    min_source = points_source.min(dim=0).values
    center = (max_source + min_source) * 0.5
    max_source = (max_source - center) * 2 + center
    min_source = (min_source - center) * 2 + center
    
    # Focus on middle portion of object (avoid base/top edges)
    min_bound_z = points_source[:, -1].min() + min(
        max((points_source[:, -1].max() - points_source[:, -1].min()) * 0.1, 0.05),
        (points_source[:, -1].max() - points_source[:, -1].min()) * 0.5
    )
    max_bound_z = points_source[:, -1].max()
    
    if max_bound_z - min_bound_z < 0.05:
        return torch.tensor([0.0]).to(points_source.device), False, None, None, 0

    # Filter target points inside expanded box
    keep_target_points = (
        (points_target[:, 0] < max_source[0].cpu()) & 
        (points_target[:, 0] > min_source[0].cpu()) &
        (points_target[:, 1] < max_source[1].cpu()) & 
        (points_target[:, 1] > min_source[1].cpu()) &
        (points_target[:, 2] > min_bound_z.cpu()) & 
        (points_target[:, 2] < max_bound_z.cpu())
    )
    
    if keep_target_points.sum() == 0:
        return torch.tensor([0.0]).to(points_source.device), False, None, None, keep_target_points.sum()
    
    k = 1 if keep_target_points.sum() > 1 else keep_target_points.sum()

    # Sample and compute distances
    points_target = points_target[keep_target_points].to(points_source.device)
    dists, ref_dist, no_target, sampled_points_source, sampled_points_target = get_distances(
        points_source, points_target, num_points, p_norm=1
    )
    
    if no_target:
        return torch.tensor([0.0]).to(points_source.device), False, None, None, 0
    
    # Compute 2D distance
    dists_2d = torch.norm(
        sampled_points_source[:, :2] - sampled_points_target[:, :2].mean(dim=0)
    )

    # Find worst collision point
    anchor_point_id = torch.argmax(dists_2d)
    
    # Check if actual collision (distance < 50% of reference distance)
    tracking_loss = (
        torch.cdist(sampled_points_source, sampled_points_target).min() < 0.5 * ref_dist 
        if keep_target_points.sum() != 0 else False
    )
    
    # Use floor boundary if provided
    if floor_cont_points_normals is not None:
        source_points_dist_to_cont = signed_distance(
            floor_cont_points_normals[0], floor_cont_points_normals[1],
            sampled_points_source[:, :2], k=1, return_mean=False
        )
        anchor_point_id = torch.argmin(source_points_dist_to_cont)
        
        target_points_dist_to_cont = signed_distance(
            floor_cont_points_normals[0], floor_cont_points_normals[1],
            sampled_points_target[:, :2], k=1, return_mean=False
        )
        anchor_point_target_id = torch.argmax(target_points_dist_to_cont)
    else:
        source_points_dist_to_cont = None

    # Penalize distances to other movable objects' centers
    if len(targets_centers) != 0:
        source_center = points_source.mean(dim=0)
        center_to_center_distance = torch.norm(
            source_center[None, :2] - targets_centers[:, :2], dim=-1, p=1
        )
        # Weight by inverse distance (closer objects matter more)
        inv_distance = 1 / center_to_center_distance.detach()
        probs = inv_distance / inv_distance.max()
        centers_loss = (torch.clamp(50 - center_to_center_distance, min=0.0) * probs).mean()
    else:
        centers_loss = 0.0
    
    # Different strategies based on floor boundary
    if source_points_dist_to_cont is not None and source_points_dist_to_cont.max() < 0:
        # All source points inside floor boundary - use relative distances
        rel_dists = dists
        collision_loss = (
            torch.clamp(50 - rel_dists, min=0.0).mean() + centers_loss 
            if tracking_loss else torch.tensor([0.0]).to(points_source.device)
        )
    else:
        # Use anchor point strategy
        collision_loss = (
            torch.clamp(50 - dists[anchor_point_id], min=0.0).mean() + centers_loss
            if tracking_loss else torch.tensor([0.0]).to(points_source.device)
        )
    
    return collision_loss, tracking_loss, None, None, len(sampled_points_target)


def find_low_density_missing_point(xy, grid_size=0.1, neighborhood_radius=1.0):
    """
    Find a low-density point in a 2D grid that doesn't exist in xy.
    
    Algorithm:
        1. Create regular grid covering xy range
        2. Find grid points NOT in xy (missing points)
        3. Count neighbors within radius for each missing point
        4. Select random point from those with minimum density
        
    Math:
        For each missing point m:
            density(m) = |{p in xy : ||p - m|| < radius}|
        Select random m where density(m) = min(density)
        
    Use case: Finding placement locations with minimum clutter
    
    Args:
        xy: Existing points (N x 2)
        grid_size: Grid cell size for sampling (meters)
        neighborhood_radius: Radius for counting neighbors (meters)
        
    Returns:
        A single low-density missing point (1 x 2)
        
    Raises:
        ValueError: If no missing points found (grid too coarse)
    """
    # Define grid range
    min_x, min_y = xy.min(dim=0).values
    max_x, max_y = xy.max(dim=0).values

    # Create grid of candidate points
    xs = torch.arange(min_x, max_x, grid_size)
    ys = torch.arange(min_y, max_y, grid_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    grid_points = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)

    # Find points that are NOT in xy
    dist_to_xy = torch.cdist(grid_points, xy)
    mask = (dist_to_xy < 1e-5).any(dim=1)  # True for points that exist in xy
    missing_points = grid_points[~mask]    # Exclude existing points

    if len(missing_points) == 0:
        raise ValueError("No missing points found in the grid. Try increasing grid_size.")

    # Compute neighbor counts for missing points (density measure)
    dist_missing_to_xy = torch.cdist(missing_points, xy)
    neighbor_counts = (dist_missing_to_xy < neighborhood_radius).sum(dim=1)

    # Find minimum density
    min_density = neighbor_counts.min()

    # Get all points with lowest density
    low_density_indices = torch.where(neighbor_counts == min_density)[0]

    # Randomly select one
    selected_index = low_density_indices[torch.randint(len(low_density_indices), (1,))]

    return missing_points[selected_index]

