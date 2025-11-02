import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import copy
from models.components.geometry.object_point_cloud import TransformationNetwork  # NEW
from models.components.optimization.losses import (
    collision_loss, on_top_of_loss, on_top_of_loss_bbox, against_loss
)
from models.components.utils.tree_utils import sort_tree_relations_with_indices
from models.components.geometry.transformations import (
    get_individual_transformations,
    rotation_matrix 
)

class SceneOptimizer:
    """Handles scene optimization for object placement"""
    
    def __init__(self, handler):
        self.handler = handler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def optimize_scene(self):
        """Main optimization loop"""
        T = self.fit_transformation()
        
        for m_id, t in zip(self.handler.trainable_objects, T):
            transformed_group = (self.handler.get_linked_groups()[m_id] + [m_id])
            for g_ele_i in transformed_group:
                self.handler.update_point_cloud(g_ele_i, t, m_id)
                
        scene_points = np.asarray(self.handler.data.vertices)
        
        # Adjust z-position based on support surface
        for m_id in self.handler.trainable_objects:
            z_shift = self._calculate_z_shift(m_id, scene_points)
            t = np.eye(4)
            t[2, -1] = z_shift
            transformed_group = self.handler.objects[m_id].supported_objects + [m_id]
            
            for g_ele_i in transformed_group:
                self.handler.update_point_cloud(g_ele_i, t, m_id)
                
        if self.handler.save_video:
            self.handler.video.release()
            
        return None
        
    def _calculate_z_shift(self, m_id, scene_points):
        """Calculate z-shift to place object on support surface"""
        object_points = scene_points[self.handler.class_agnostic_masks[m_id]]
        other_points = scene_points[~self.handler.class_agnostic_masks[m_id]]
        max_point = object_points[:, :2].max(axis=0)
        min_point = object_points.min(axis=0)
        
        other_points_in_box = other_points[
            (np.all(other_points[:, :2] < max_point, axis=-1) *
             np.all(other_points[:, :2] > min_point[:2], axis=-1) *
             (other_points[:, 2] < min_point[2] + 0.1) *
             (other_points[:, 2] >= min_point[2] - 0.1)) == 1
        ]
        
        try:
            other_points_in_box = self._filter_support_surfaces(other_points_in_box)
        except:
            try:
                floor_z = scene_points[:, 2].min()
                mean_p = object_points[:, :2].mean(axis=0)
                max_point = (max_point - mean_p) * 2 + mean_p
                min_point[:2] = (min_point[:2] - mean_p) * 2 + mean_p
                other_points_in_box = other_points[
                    (np.all(other_points[:, :2] < max_point, axis=-1) *
                     np.all(other_points[:, :2] > floor_z - 0.1, axis=-1) *
                     (other_points[:, 2] < min_point[2] + 0.1) *
                     (other_points[:, 2] >= min_point[2] - 0.1)) == 1
                ]
                other_points_in_box = self._filter_support_surfaces(other_points_in_box)
            except:
                floor_z = scene_points[:, 2].min()
                other_points_in_box = np.array([[0, 0, floor_z]])
                
        z_shift = (other_points_in_box[:, 2].max() - min_point[2] 
                   if len(other_points_in_box) != 0 else 0)
        return z_shift
        
    @staticmethod
    def _filter_support_surfaces(other_points_in_box, bin_size=0.03):
        """Filter support surfaces based on z-distribution"""
        z_dist = other_points_in_box[:, 2]
        hist, bin_edges = np.histogram(
            z_dist,
            bins=np.arange(z_dist.min(), z_dist.max() + bin_size, bin_size)
        )
        
        top_3_bin_indices = np.argsort(hist)[-3:][::-1]
        closest_bin_idx = min(top_3_bin_indices)
        z_min, z_max = bin_edges[closest_bin_idx], bin_edges[closest_bin_idx + 1]
        
        support_mask = (z_dist >= z_min) & (z_dist < z_max)
        return other_points_in_box[support_mask]
        
    def fit_transformation(self):
        """Fit transformations for trainable objects"""
        loss = self.group_optimization(num_epochs=100, initilize=True)
        loss = self.constraints_optimization()
        
        if True:
            return [t.get_transformation() for t in self.Ts]
        else:
            print(f"loss {loss} is too high, instruction cannot be executed")
            return [np.eye(4) for t in self.Ts]
            
    def group_optimization(self, num_epochs=2, device='cuda', lr=1.,
                          detach_theta=True, initilize=False, t_precision=4):
        """
        Optimize object groups
        
        Args:
            num_epochs: Number of optimization epochs
            device: Device to use
            lr: Learning rate
            detach_theta: Whether to detach theta gradient
            initilize: Whether to initialize networks
            t_precision: Precision for theta steps
            
        Returns:
            Whether optimization failed
        """
        from collections import Counter
        
        def process_data(data):
            result = {}
            for key, tuples in data.items():
                counter = Counter([t[0] for t in tuples])
                most_common_id = counter.most_common(1)[0][0]
                avg_value = sum(t[1] for t in tuples if t[0] == most_common_id) / len(
                    [t for t in tuples if t[0] == most_common_id]
                )
                result[key] = (most_common_id, avg_value)
            return result
            
        # Process maintain distance constraints
        self.handler.maintain_group_distance = {}
        for g_i, g in enumerate(self.handler.trainable_groups):
            for i in g:
                if (i in self.handler.maintain_distance.keys() and
                    int(self.handler.maintain_distance[i][0]) not in g):
                    if g_i not in self.handler.maintain_group_distance.keys():
                        self.handler.maintain_group_distance[g_i] = []
                    self.handler.maintain_group_distance[g_i].append(
                        self.handler.maintain_distance[i]
                    )
        self.handler.maintain_group_distance = process_data(
            self.handler.maintain_group_distance
        )
        
        if initilize:
            self.Tg = [TransformationNetwork() 
                      for _ in range(len(self.handler.trainable_groups))]
            params = []
            for network in self.Tg:
                params += list(network.parameters())
            self.optimizer = optim.SGD(params, lr=lr)
            
            self.apply_trainable_transf_to = self.handler.get_linked_groups()
            for obj_id, g in enumerate(self.handler.get_linked_groups()):
                for item in g:
                    if item in self.handler.trainable_objects:
                        self.apply_trainable_transf_to[obj_id].remove(item)
                        
        # Collect group IDs
        collect_group_ids = []
        for group in self.handler.trainable_groups:
            curr_group = group if self.handler.objects[group[0]].train else group[1:]
            collect_group_ids.append(curr_group)
            for obj_id in curr_group:
                collect_group_ids[-1] += self.apply_trainable_transf_to[obj_id]
                
        # Store first stage transformations
        self.handler.first_stage_id_to_transform = {}
        for g_id, g in enumerate(collect_group_ids):
            transformations_list = get_individual_transformations(
                [self.handler.objects[id].points for id in g],
                self.Tg[g_id].x.item(),
                self.Tg[g_id].y.item(),
                self.Tg[g_id].theta.item()
            )
            for obj_id, trasn_dict in zip(g, transformations_list):
                if self.handler.objects[obj_id].train:
                    self.handler.first_stage_id_to_transform[obj_id] = trasn_dict
                    
        return False
        
    def constraints_optimization(self, num_epochs=300, device='cuda', lr=1):
        """
        Optimize with constraints
        
        Args:
            num_epochs: Number of epochs
            device: Device to use
            lr: Learning rate
            
        Returns:
            Final loss value
        """
        # Get support surfaces for group items
        support_surfaces_for_group_items = []
        
        def get_surface(id):
            for c in self.handler.objects[id].constraints:
                if c[0] == 'on top of':
                    s_id, o_id = c[-1], c[-2]
                    return [o_id, s_id]
            s_id, o_id = 0, self.handler.floor_id
            return [o_id, s_id]
            
        for member_id in self.handler.trainable_objects:
            s_obj_id, s_surf_id = get_surface(member_id)
            for g_id, g in enumerate(self.handler.trainable_groups):
                if member_id in g:
                    group = g_id
                    break
            support_surfaces_for_group_items.append((s_obj_id, s_surf_id, group))
            
        # Initialize transformation networks
        self.Ts = []
        for objl_id, id in enumerate(self.handler.trainable_objects):
            x_init = self.handler.first_stage_id_to_transform[id]['x']
            y_init = self.handler.first_stage_id_to_transform[id]['y']
            theta_init = self.handler.first_stage_id_to_transform[id]['theta']
            self.Ts.append(
                TransformationNetwork(x_init=x_init, y_init=y_init, theta_init=theta_init)
            )
            
        params = []
        for param in [network.parameters() for network in self.Ts]:
            params += list(param)
        self.optimizer = optim.SGD(params, lr=lr)
        
        # Setup
        scene_points = torch.from_numpy(np.array(self.handler.data.vertices))
        self.apply_trainable_transf_to = self.handler.get_linked_groups()
        for obj_id, g in enumerate(self.handler.get_linked_groups()):
            for item in g:
                if item in self.handler.trainable_objects:
                    self.apply_trainable_transf_to[obj_id].remove(item)
                    
        obj_to_group = {}
        for group in self.handler.trainable_groups:
            for i in group:
                obj_to_group[i] = group
                
        ratio = 4
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=20, eta_min=1e-6
        )
        decay_factor = 0.99
        
        # Training loop
        for epoch in range(num_epochs):
            mat_weight = (decay_factor ** epoch)
            
            for network in self.Ts:
                network.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            pc_in = [self.handler.objects[id].points_torch 
                    for id in self.handler.trainable_objects]
            pc_out = [net(pc_in_, detach_theta=True) 
                     for net, pc_in_ in zip(self.Ts, pc_in)]
            
            # Distance matrix initialization
            if epoch == 0:
                n = len(self.handler.trainable_objects)
                weight_matrix = np.zeros((n, n), dtype=int)
                
                for i in range(n):
                    for j in range(n):
                        if (i != j and 
                            support_surfaces_for_group_items[i] == 
                            support_surfaces_for_group_items[j]):
                            weight_matrix[i, j] = 1
                            
                objects_centroids = torch.stack([pc_i.mean(dim=0) for pc_i in pc_out])
                distances = []
                for pc_i in pc_out:
                    distances.append([])
                    for pc_i_o in pc_out:
                        distances[-1].append(
                            torch.cdist(pc_i[:500], pc_i_o[:500]).min()
                        )
                distances = torch.tensor(distances)
                gt_vecs = (objects_centroids[:, None, :] - 
                          objects_centroids[None, :, :]).detach()
                inverted_distance = distances - 0.001
                inverted_distance = 1 / inverted_distance
                inverted_distance = (inverted_distance / 
                                    inverted_distance.max(dim=-1).values[:, None])
                weight_matrix = (torch.from_numpy(weight_matrix).to(inverted_distance.device) * 
                               inverted_distance)
                weight_matrix = weight_matrix[:, :, None]
                
            # Group forward pass
            pc_group_in = [
                torch.cat([self.handler.objects[id].points_torch 
                          for id in [m_i] + self.apply_trainable_transf_to[m_i]])
                for m_i in self.handler.trainable_objects
            ]
            
            pc_group_out_xy = [
                net(pc_in_, pc_out[id].mean(dim=0).detach(), detach_theta=True)
                for id, (net, pc_in_) in enumerate(zip(self.Ts, pc_group_in))
            ]
            
            # Calculate losses
            bs = len(pc_group_out_xy)
            loss_constraint, loss_collision, in_boundry_loss = [], [], []
            
            # Collision loss
            for obj_id in range(bs):
                loss_collision.append([])
                
                other_tranable_in_surface = []
                source_surface = get_surface(self.handler.trainable_objects[obj_id])
                
                for trainable_id in self.handler.trainable_objects:
                    if (trainable_id == self.handler.trainable_objects[obj_id] or
                        trainable_id not in obj_to_group[self.handler.trainable_objects[obj_id]]):
                        continue
                    other_obj_surface = get_surface(trainable_id)
                    if source_surface == other_obj_surface:
                        id_o = self.handler.trainable_objects.index(trainable_id)
                        other_tranable_in_surface.append(pc_group_out_xy[id_o])
                        
                # Filter masks
                filter_out_masks = [self.handler.class_agnostic_masks[source_surface[0]]]
                filter_out_masks += [
                    m for c, m in zip(self.handler.class_names, 
                                     self.handler.class_agnostic_masks)
                    if c in ['wall', 'floor']
                ]
                
                for m_i in self.handler.trainable_objects:
                    for m_ig in [m_i] + self.apply_trainable_transf_to[m_i]:
                        filter_out_masks.append(self.handler.class_agnostic_masks[m_ig])
                        
                mask = sum(filter_out_masks)
                compl_scene = scene_points[mask == 0]
                compl_scene_with_train = (
                    torch.cat([compl_scene.to(other_tranable_in_surface[0].device)] + 
                             other_tranable_in_surface)
                    if len(other_tranable_in_surface) > 0
                    else compl_scene
                )
                
                other_trainable_centers = (
                    torch.stack([o.mean(dim=0).detach() for o in other_tranable_in_surface])
                    if other_tranable_in_surface
                    else []
                )
                
                if self.handler.objects[source_surface[0]].surfaces:
                    surface_points = self.handler.objects[source_surface[0]].surfaces[
                        source_surface[1]
                    ]['contour points']
                    surface_normals = self.handler.objects[source_surface[0]].surfaces[
                        source_surface[1]
                    ]['contour normals']
                    surface_points = (torch.from_numpy(surface_points).to(device)
                                    if isinstance(surface_points, np.ndarray)
                                    else surface_points.to(device))
                    surface_normals = (torch.from_numpy(surface_normals).to(device)
                                     if isinstance(surface_normals, np.ndarray)
                                     else surface_normals.to(device))
                    surface_info = (surface_points, surface_normals)
                else:
                    surface_info = None
                    
                loss_col_i, tracking_loss, _, _, _ = collision_loss(
                    pc_group_out_xy[obj_id], compl_scene_with_train,
                    other_trainable_centers, 2500,
                    floor_cont_points_normals=surface_info
                )
                
                if tracking_loss:
                    loss_collision[-1].append(loss_col_i)
                else:
                    loss_collision[-1].append(loss_col_i * 0.0)
                    
            # Constraint losses
            for obj_id in self.handler.trainable_objects:
                support_s_found = False
                for c in self.handler.objects[obj_id].constraints:
                    on_top_base_th = 0.1
                    
                    if c[0] == 'on top of':
                        surface_scale = 0.7 if c[1] != 'floor' else 1.
                        loss_constraint.append(
                            self._calculate_on_top_loss(
                                obj_id, c, surface_scale, on_top_base_th, 
                                pc_group_out_xy, pc_out
                            )
                        )
                        support_s_found = True
                        
                    if c[0] == 'against':
                        loss_constraint.append(
                            self._calculate_against_loss(obj_id, c, pc_group_out_xy)
                        )
                        
                if not support_s_found:
                    # Default to floor
                    surface_points = self.handler.objects[self.handler.floor_id].surfaces[0]['contour points']
                    surface_normals = self.handler.objects[self.handler.floor_id].surfaces[0]['contour normals']
                    surface_elevation = self.handler.objects[self.handler.floor_id].surfaces[0]['elevation']
                    surface_points = (torch.from_numpy(surface_points).to(device)
                                    if isinstance(surface_points, np.ndarray)
                                    else surface_points.to(device))
                    surface_normals = (torch.from_numpy(surface_normals).to(device)
                                     if isinstance(surface_normals, np.ndarray)
                                     else surface_normals.to(device))
                    surface_elevation = (torch.from_numpy(surface_elevation).to(device)
                                       if isinstance(surface_elevation, np.ndarray)
                                       else surface_elevation.to(device))
                    loss_constraint.append(
                        on_top_of_loss(
                            surface_points[:, :2], surface_normals[:, :2],
                            surface_elevation[-1],
                            pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1].min(),
                            pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, :2],
                            1.0, 0.5
                        )[0]
                    )
                    
            # Combine losses
            loss = torch.tensor(0.0).to(device)
            col_loss_item = torch.tensor(0.0).to(device)
            in_boundry_loss_item = torch.tensor(0.0).to(device)
            const_loss_item = torch.tensor(0.0).to(device)
            
            if loss_constraint:
                loss += 5 * sum(loss_constraint)
                const_loss_item = sum(loss_constraint).item()
                
            if loss_collision:
                for col_los in loss_collision:
                    if col_los:
                        meas_loss = sum(col_los) / len(col_los)
                        col_loss_item += meas_loss[0] if meas_loss.dim() == 1 else meas_loss
                loss += col_loss_item
                col_loss_item = col_loss_item.item()
                
            if in_boundry_loss:
                loss += sum(in_boundry_loss)
                in_boundry_loss_item = sum(in_boundry_loss).item()
                
            # Group loss
            objects_centroids = torch.stack([pc_i.mean(dim=0) for pc_i in pc_out])
            curr_vecs = (objects_centroids[:, None, :] - objects_centroids[None, :, :])
            gt_vecs_norm = torch.norm(gt_vecs, dim=-1, keepdim=True)
            curr_vecs_norm = torch.norm(curr_vecs, dim=-1, keepdim=True)
            
            gt_vec_normalizer = 1
            curr_vec_normalizer = 1
            
            loss += torch.norm(
                (gt_vecs / gt_vec_normalizer - curr_vecs / curr_vec_normalizer) *
                ((weight_matrix).to(curr_vecs.device))
            )
            
            # Maintain distance loss
            def get_points(idx):
                if idx in self.handler.trainable_objects:
                    return pc_out[self.handler.trainable_objects.index(idx)]
                else:
                    return self.handler.objects[idx].points_torch.to(device)
                    
            center_loss = []
            for k, v in self.handler.maintain_distance.items():
                source_dims = self.handler.objects[k].get_obj_details()['dimensions']
                center_dims = self.handler.objects[int(v[0])].get_obj_details()['dimensions']
                margin = (min(source_dims[:2]) + min(center_dims[:2])) / 2 + 1
                distance = torch.norm(
                    torch.norm(get_points(k)[:, :2].mean(dim=0) - 
                             get_points(int(v[0]))[:, :2].mean(dim=0).detach())
                )
                center_loss.append(
                    torch.norm(distance - v[1]) *
                    ((distance.detach() < v[1] + margin) *
                     (distance.detach() > v[1] - margin) == 0)
                )
            loss += 0.2 * sum(center_loss)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            for obj_loc_id, t in enumerate(self.Ts):
                obj_dims = self.handler.objects[
                    self.handler.trainable_objects[obj_loc_id]
                ].get_obj_details(get_position_and_orientation=True, in_obj_frame=True)
                x_step = max(obj_dims['dimensions'][0] / ratio, 0.05)
                y_step = max(obj_dims['dimensions'][1] / ratio, 0.05)
                x_grad_norm = torch.norm(t.x.grad)
                y_grad_norm = torch.norm(t.y.grad)
                
                if x_grad_norm == 0 and y_grad_norm == 0:
                    continue
                    
                max_dir = torch.max(abs(t.x.grad), abs(t.y.grad))
                t.x.grad = (x_step / lr) * (t.x.grad / max_dir) * self.optimizer.param_groups[0]['lr']
                t.y.grad = (y_step / lr) * (t.y.grad / max_dir) * self.optimizer.param_groups[0]['lr']
                
            # Handle hierarchical constraints
            sorted_trainable_object_to_support, _ = sort_tree_relations_with_indices(
                list(self.handler.trainable_object_to_support.items())
            )
            relevent_transformations = [
                (self.handler.trainable_objects.index(indx[0]),
                 self.handler.trainable_objects.index(indx[1]))
                for indx in sorted_trainable_object_to_support
                if (indx[0] in self.handler.trainable_objects and
                    indx[1] in self.handler.trainable_objects)
            ]
            
            for supported_id, support_id in relevent_transformations:
                T_support = self.Ts[support_id]
                
                center_to_center_vec = (
                    pc_out[supported_id].mean(dim=0) - 
                    pc_out[support_id].mean(dim=0)
                ).detach().cpu().numpy()
                rel_x = center_to_center_vec[0]
                rel_y = center_to_center_vec[1]
                
                cos_t, sin_t = np.cos(T_support.theta.grad.item()), np.sin(T_support.theta.grad.item())
                drel_x_new = cos_t * rel_x - sin_t * rel_y - rel_x
                drel_y_new = sin_t * rel_x + cos_t * rel_y - rel_y
                
                t.x.grad = drel_x_new + T_support.x.grad + t.x.grad
                t.y.grad = drel_y_new + T_support.y.grad + t.y.grad
                t.theta.grad = T_support.theta.grad + t.theta.grad
                
            # Update parameters
            self.optimizer.step()
            scheduler.step()
            
            # Logging
            if epoch % 1 == 0:
                groups = []
                for obj_id in self.handler.trainable_objects:
                    groups.append([obj_id] + self.apply_trainable_transf_to[obj_id])
                if self.handler.save_video:
                    self.handler.create_video(
                        [t.get_transformation() for t in self.Ts],
                        groups, text=f"Object optimization", epoch=epoch
                    )
                self.handler.log_pose(
                    [t.get_transformation() for t in self.Ts],
                    groups, epoch=epoch, append_suffix_to_path='_second_stage'
                )
                
            # Early stopping
            if loss.item() < 0.005:
                if epoch % 1 == 0:
                    groups = []
                    for obj_id in self.handler.trainable_objects:
                        groups.append([obj_id] + self.apply_trainable_transf_to[obj_id])
                    if self.handler.save_video:
                        self.handler.create_video(
                            [t.get_transformation() for t in self.Ts],
                            groups, text=f"Object optimization", epoch=epoch
                        )
                    self.handler.log_pose(
                        [t.get_transformation() for t in self.Ts],
                        groups, epoch=epoch, append_suffix_to_path='_second_stage'
                    )
                return loss.item()
                
            if epoch == num_epochs - 1:
                if epoch % 1 == 0:
                    groups = []
                    for obj_id in self.handler.trainable_objects:
                        groups.append([obj_id] + self.apply_trainable_transf_to[obj_id])
                    if self.handler.save_video:
                        self.handler.create_video(
                            [t.get_transformation() for t in self.Ts],
                            groups, text=f"Object optimization", epoch=epoch
                        )
                    self.handler.log_pose(
                        [t.get_transformation() for t in self.Ts],
                        groups, epoch=epoch, append_suffix_to_path='_second_stage'
                    )
                return 0.0
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
                print(f'Constraint Loss: {const_loss_item:.4f}')
                print(f'Collision Loss: {col_loss_item:.4f}')
                print(f'In Boundary Loss: {in_boundry_loss_item:.4f}')
                
        return 0.0
        
    def _calculate_on_top_loss(self, obj_id, constraint, surface_scale, 
                               on_top_base_th, pc_group_out_xy, pc_out):
        """Calculate on top of constraint loss"""
        device = self.device
        
        if constraint[2] in self.handler.trainable_objects:
            if len(self.handler.objects[constraint[2]].surfaces) > 0:
                surface_points = self.handler.objects[constraint[2]].surfaces[
                    constraint[-1]
                ]['contour points']
                surface_normals = self.handler.objects[constraint[2]].surfaces[
                    constraint[-1]
                ]['contour normals']
                surface_elevation = self.handler.objects[constraint[2]].surfaces[
                    constraint[-1]
                ]['elevation']
                
                surface_points = (torch.from_numpy(surface_points).to(device)
                                if isinstance(surface_points, np.ndarray)
                                else surface_points.to(device))
                surface_normals = (torch.from_numpy(surface_normals).to(device)
                                 if isinstance(surface_normals, np.ndarray)
                                 else surface_normals.to(device))
                surface_elevation = (torch.from_numpy(surface_elevation).to(device)
                                   if isinstance(surface_elevation, np.ndarray)
                                   else surface_elevation.to(device))
                
                centroid = surface_points.mean(dim=0)
                surface_points = (surface_points - centroid) * surface_scale + centroid
                
                transformated_contour = self.Ts[
                    self.handler.trainable_objects.index(constraint[2])
                ](surface_points, center=pc_out[
                    self.handler.trainable_objects.index(constraint[2])
                ].mean(dim=0)).detach()[:, :2]
                
                transformated_normals = self.Ts[
                    self.handler.trainable_objects.index(constraint[2])
                ].forward_xy_rotate(surface_normals).detach()[:, :2]
                
                transformated_elevation = self.Ts[
                    self.handler.trainable_objects.index(constraint[2])
                ].forward_z(surface_elevation).detach().squeeze()[-1]
                
                mask = (torch.ones_like(
                        pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1]
                    ) == 1 if constraint[1] == 'floor'
                        else pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1] < 
                                pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1].min() +
                                on_top_base_th)
                
                return on_top_of_loss(
                    transformated_contour, transformated_normals, transformated_elevation,
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1].min(),
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][mask][:, :2],
                    1.0, 4.0
                )[0]
            else:
                return on_top_of_loss_bbox(
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)],
                    pc_out[self.handler.trainable_objects.index(constraint[2])].detach()
                )
        else:
            if len(self.handler.objects[constraint[2]].surfaces) > 0:
                surface_points = self.handler.objects[constraint[2]].surfaces[
                    constraint[-1]
                ]['contour points']
                surface_normals = self.handler.objects[constraint[2]].surfaces[
                    constraint[-1]
                ]['contour normals']
                surface_elevation = self.handler.objects[constraint[2]].surfaces[
                    constraint[-1]
                ]['elevation']
                
                surface_points = (torch.from_numpy(surface_points).to(device)
                                if isinstance(surface_points, np.ndarray)
                                else surface_points.to(device))
                surface_normals = (torch.from_numpy(surface_normals).to(device)
                                 if isinstance(surface_normals, np.ndarray)
                                 else surface_normals.to(device))
                surface_elevation = (torch.from_numpy(surface_elevation).to(device)
                                   if isinstance(surface_elevation, np.ndarray)
                                   else surface_elevation.to(device))
                
                centroid = surface_points.mean(dim=0)
                surface_points = (surface_points - centroid) * surface_scale + centroid
                
                mask = (torch.ones_like(
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1]
                ) == 1 if constraint[1] == 'floor'
                    else pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1] < 
                            pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1].min() +
                            on_top_base_th)
                
                return on_top_of_loss(
                    surface_points[:, :2], surface_normals[:, :2], surface_elevation[-1],
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][:, -1].min(),
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)][mask][:, :2],
                    1.0, 0.5
                )[0]
            else:
                return on_top_of_loss_bbox(
                    pc_group_out_xy[self.handler.trainable_objects.index(obj_id)],
                    self.handler.objects[constraint[2]].points_torch
                )[0]
                
    def _calculate_against_loss(self, obj_id, constraint, pc_group_out_xy):
        """Calculate against wall constraint loss"""
        device = self.device
        
        wall_details = self.handler.objects[int(constraint[-1])].get_obj_details(
            get_position_and_orientation=True
        )
        abs_orientation = wall_details['oientation']
        roatation_center = torch.tensor(wall_details['base']).to(device)
        R = torch.from_numpy(
            rotation_matrix(np.array([0, 0, 1]), np.radians(abs_orientation))
        ).to(device)
        
        wall_points_in_frame = (
            self.handler.objects[int(constraint[-1])].points_torch.to(device) - 
            roatation_center[None, :]
        ) @ R
        
        out_points_in_wall_frame = (
            pc_group_out_xy[self.handler.trainable_objects.index(obj_id)] - 
            roatation_center[None, :]
        ) @ R
        
        dx = (out_points_in_wall_frame[:, 0].max() - 
              out_points_in_wall_frame[:, 0].min()).detach()
        dx_wall = (wall_points_in_frame[:, 0].max() - 
                  wall_points_in_frame[:, 0].min()).detach()
        dy_wall = (wall_points_in_frame[:, 1].max() - 
                  wall_points_in_frame[:, 1].min()).detach()
        
        y_center = out_points_in_wall_frame[:, 1].mean()
        y_max = out_points_in_wall_frame[:, 1].max()
        y_min = out_points_in_wall_frame[:, 1].min()
        
        in_walls_dy = ((y_max < dy_wall / 2) * (y_max > -dy_wall / 2) *
                      (y_min < dy_wall / 2) * (y_min > -dy_wall / 2))
        
        distance_to_wall = ((dx + dx_wall) / 2 - 
                           (out_points_in_wall_frame[:, 0].max() + 
                            out_points_in_wall_frame[:, 0].min()) / 2)
        
        against_loss = (torch.norm(distance_to_wall) + 
                       torch.norm(y_center) * (in_walls_dy == 0))
        
        return against_loss