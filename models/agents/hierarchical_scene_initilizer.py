
import xmltodict
from models.components.utils.text_utils import extract_text_between_tags
from models.agents.prompts.hierarchical_scene_initializer import get_prompt, get_group_refinement_prompt
from models.components.utils.io_utils import load_or_execute
from models.components.utils.text_utils import parse_and_compute
import numpy as np
import os
import torch
import copy
from collections import Counter
import xmltodict
from models.components.utils.io_utils import load_yaml, save_yaml
from models.components.geometry.transformations import rotation_matrix as get_rotation_matrix
from collections import defaultdict

def flatten_hierarchy(node, parent={}, grandparent={}, grand_grand_parent={}):
    flattened = []
    
    if isinstance(node, dict):
        name = node.get("name")
        parent_name = parent.get("name") if parent else None
        parent_id = parent.get("id") if parent else None
        
        grand_parent_name = grandparent.get("name") if grandparent else None
        grand_parent_id = grandparent.get("id") if grandparent else None
        
        grand_grand_parent_name = grand_grand_parent.get("name") if grand_grand_parent else None
        grand_grand_parent_id = grand_grand_parent.get("id") if grand_grand_parent else None

        # Adjust node ID if it's a surface
        if name == 'surface':
            node_id = f"{parent_id}_{node['id']}"
        else:
            node_id = node["id"]

        # Adjust parent ID if parent is a surface
        if parent_name == 'surface':
            parent_id = f"{grand_parent_id}_{parent_id}"
        
        # Adjust grandparent ID if grandparent is a surface
        if grand_parent_name == 'surface':
            grand_parent_id = f"{grand_grand_parent_id}_{grand_parent_id}"

        flattened.append({
            "name": name,
            "id": node_id,
            "instruction": node.get("instruction"),
            "parent_name": parent_name,
            "parent_id": parent_id,
            "grand_parent_name": grand_parent_name,
            "grand_parent_id": grand_parent_id,
            "grand_grand_parent_name": grand_grand_parent_name,
            "grand_grand_parent_id": grand_grand_parent_id
        })

        # Recursively process children
        children = node.get("object")
        if children:
            if isinstance(children, list):
                for child in children:
                    flattened.extend(flatten_hierarchy(child, parent=node, grandparent=parent, grand_grand_parent=grandparent))
            else:
                flattened.extend(flatten_hierarchy(children, parent=node, grandparent=parent, grand_grand_parent=grandparent))

    return flattened
def map_angle_to_360(angle):
    """
    Map an angle from the range 0 to -180 (or 0 to 180) to the 0 to 360 convention.
    
    Parameters:
    angle (float): The angle in degrees.
    
    Returns:
    float: The angle mapped to the 0 to 360 range.
    """
    # If the angle is negative, add 360 to bring it into the 0 to 360 range
    if angle < 0:
        angle += 360
    return angle

def angle_between_vectors(vector):
    """
    Calculate the angle in radians from the reference_vector to the vector using arctan2.
    
    Parameters:
    vector (tuple or list): The target vector as (x, y).
    reference_vector (tuple or list): The reference vector as (x, y).
    
    Returns:
    float: The angle in radians.
    """
    # Convert the vectors to numpy arrays
    # Calculate the difference vector
    difference_vector = vector
    
    # Calculate the angle using arctan2
    angle = np.arctan2(difference_vector[1], difference_vector[0])
    
    return np.degrees(angle)
def get_adjusted_orientation(old_orientation, member_relative_location, group_center_location):
    target_to_member_direction = np.array(group_center_location[:2])-np.array(member_relative_location[:2])
    expected_orientation = angle_between_vectors(target_to_member_direction)
    adjusted_orientation = expected_orientation if abs(map_angle_to_360(expected_orientation)-map_angle_to_360(old_orientation)) > 45 else old_orientation
    return adjusted_orientation
def adjust_orientation(obj, set_to=None):
    new_obj = copy.deepcopy(obj)
    old_orientation = parse_and_compute(obj['new_orientation'])[0]
    new_base_coordinate = np.array(parse_and_compute(obj['new_base_coordinate']))
    if np.all(new_base_coordinate[:2] == 0):
        return obj
    new_obj['new_orientation'] = str(get_adjusted_orientation(old_orientation, new_base_coordinate[:2], np.array([0,0]))) if set_to == None else str(set_to)
    return new_obj

def constraint_to_orientation_consistency(list_of_objects_orientation_refined):
    list_of_objects_orientation_refined_corrected = []
    # orientations_dist = np.array([parse_and_compute(i['new_orientation'])[0] for i in list_of_objects_orientation_refined])
    # set_to = 180 if np.var(orientations_dist) < 10 else None
    set_to = None
    for obj in list_of_objects_orientation_refined:
        if 'yes' in obj['constraints']['facing'].lower():
            list_of_objects_orientation_refined_corrected.append(adjust_orientation(obj, set_to=set_to))
            continue
        list_of_objects_orientation_refined_corrected.append(obj)
    return list_of_objects_orientation_refined_corrected
def group_by_parent(flattened_data):
    grouped = {}
    hierarchy = {}
    for obj in flattened_data:
        parent_id = obj["parent_id"]
        
        if 'floor' not in obj["parent_name"]:
            try:
                parent_id_check = int(parent_id)
                # print(parent_id)
            except:
                obj["parent_id"] = obj["grand_parent_id"]
                # print(obj["parent_name"])
                
                obj["parent_name"] = obj["grand_parent_name"]
                obj["grand_parent_id"] = None
                # print(obj["parent_name"])
                
                obj["grand_parent_name"] = None
                # print(obj["parent_name"])
                parent_id = obj["parent_id"]
        try:
            child_id_test = int(obj["id"])
        except:
            continue        
        if parent_id not in grouped:
            grouped[parent_id] = {"root object": {'name': obj["parent_name"], 'id': obj["parent_id"], 'grand_name': obj["grand_parent_name"], 'grand_id': obj["grand_parent_id"]}, "children": []}

        # if obj["parent_name"] is None:
        #     grouped[parent_id]["root_object"] = obj
        # else:
        #     grouped[parent_id]["children"].append(obj)
        grouped[parent_id]["children"].append(obj)
        if obj["id"] is not None:
            hierarchy[obj["id"]] = obj["parent_id"]

    return grouped
def get_instruction_queue(hierarchy_dictionary):
    # ids_instruction_list = []
    try:
        flattened_data = flatten_hierarchy(hierarchy_dictionary['object'])
    
        new_data = [] 
        for data in flattened_data:
            if 'untouched' not in data['instruction'].lower() and 'leave' not in data['instruction'].lower() and 'surface' not in data['name']:
                new_data.append(data)
        
        grouped_data = group_by_parent(new_data)
        groups_list = list(grouped_data.values())
        ordered_grouped_data = build_hierarchy(groups_list)
        return ordered_grouped_data
    except:
        print('failed')
        return []

def build_hierarchy(flattened_data):
    # Step 1: Group objects by their parent_id
    tree = defaultdict(list)
    root_objects = []
    object_map = {obj["root object"]["id"]: obj for obj in flattened_data}  # Map for quick lookup

    for obj in flattened_data:
        parent_id = obj["root object"]["grand_id"]
        if parent_id is None or parent_id not in object_map:
            root_objects.append(obj)  # Treat as root if parent is missing
        else:
            tree[parent_id].append(obj)  # Maintain insertion order for children

    # Step 2: Perform hierarchical sorting using DFS
    sorted_list = []

    def traverse(node):
        sorted_list.append(node)  # Add the current node to the sorted list
        # Traverse children in the order they were added (no sorting unless needed)
        for child in tree.get(node["root object"]["id"], []):
            traverse(child)

    # Traverse root objects in the order they were added (no sorting unless needed)
    for root in root_objects:
        traverse(root)

    return sorted_list

def parse_instruction_queue(llm_response, valid_ids):
    try:
        object_new_coords_initial_text = llm_response.split('```')[1].replace('xml','')
    except:
        object_new_coords_initial_text = '<objects>'+extract_text_between_tags(llm_response.replace('xml',''), 'objects')[0]+'</objects>'

    object_new_coords_initial = xmltodict.parse(object_new_coords_initial_text)
    
    if 'objects' in object_new_coords_initial.keys():
        list_of_objects = object_new_coords_initial['objects']
    if list_of_objects is None:
        return []
    if 'object' in list_of_objects.keys():
        list_of_objects = list_of_objects['object']
    if isinstance(list_of_objects, dict):
        list_of_objects = [list_of_objects]
    
    list_of_objects = [{k.lower(): v for k, v in dictio.items()} for dictio in list_of_objects]
    pruned_list_objects = []
    for object in list_of_objects:
        try:
            if int(object['id']) in valid_ids:
                pruned_list_objects.append(object)
        except:
            continue
    
    return pruned_list_objects
def against_wall_distance(list_of_objects_orientation_refined, id_to_dim):
    list_of_objects_orientation_refined_corrected = []
    for obj in list_of_objects_orientation_refined:
        base_coords = parse_and_compute(obj['new_base_coordinate'])
        base_coords[0] = id_to_dim[int(obj['id'])][0]/2
        obj['new_base_coordinate'] = str(base_coords)
        obj['new_orientation'] = str(0)
        list_of_objects_orientation_refined_corrected.append(obj)
    return list_of_objects_orientation_refined_corrected
def Initializer(pc_handler, hierarchy_dictionary, instruction, root_path, llm = None, load_prev=True):
    if llm is None:
        exit()
    pc_handler.data = pc_handler.data_full
    for obj in pc_handler.objects:
        obj.train = False
    instruction_queue = get_instruction_queue(hierarchy_dictionary)
    pc_handler.trainable_object_to_support = {}
    pc_handler.maintain_distance = {}
    pc_handler.object_id_to_relative_details = {}
    facing_constraints = {}
    pc_handler.initlize_dummy_data(3)
    skip = []
    exclude_ids = [int(c['id']) for instruction in instruction_queue for c in instruction['children'] if 'group' not in c['id']]
    # dependency_map = {}
    # for inst in instruction_queue:
    #     for c in inst['children']:
    #         dependency_map[int(c['id'])] = int(inst['root object']['id'].split('_')[0])
    for i, objects_to_be_placed in enumerate(instruction_queue):
        
        if not objects_to_be_placed['children']:
            continue
        valid_ids = [int(child['id']) for child in objects_to_be_placed['children'] if 'group' not in child['id']]

        floor_notice = "Very Important when placing objects relative to the floor:\n Please make sure that the placement requirements are satisfied as mentioned in the instruction and also the placed objects are contained within the floor dimensions. " if objects_to_be_placed['root object']['name'] == 'floor' else ''
        try:
            prompt, inverse_transform, reference, floor_details, objects_to_be_placed, floor_xy_dims, floor_elevation = get_prompt(pc_handler, objects_to_be_placed, instruction, floor_notice=floor_notice, exclude_ids=exclude_ids)
        except:
            continue
        id_to_dim = {}
        for c in objects_to_be_placed['children']:
            id_to_dim[int(c['id'])] = c['dimensions']
        inverse_rotation = get_rotation_matrix(np.array([0,0,1]), np.radians(inverse_transform['angle_z']))
        answer = load_or_execute(root_path, f'instruction_in_queue_{i}.yaml',load_prev, llm, prompt)

        inverse_translation = inverse_transform['translation']
        list_of_objects = parse_instruction_queue(answer, valid_ids)
        if len(list_of_objects) == 0:
            continue
        reference_object_name = objects_to_be_placed['root object']['name']
        reference_object_id = objects_to_be_placed['root object']['id']
        if 'floor' in reference_object_name:
            reference_object_name = 'floor'
            reference_object_id = pc_handler.floor_id
        surface_id = None
        if isinstance(reference_object_id, str) and 'group' in reference_object_id:
            parent_id = objects_to_be_placed['root object']['grand_id'].split('_')
            grand_name = objects_to_be_placed['root object']['grand_name']
            if 'floor' in parent_id:
                surface_id = 0
                support_obj_id = pc_handler.floor_id
                support_obj_name = 'floor'
            else:
                if len(parent_id) == 1:
                    surface_found = False
                    for c in pc_handler.objects[int(parent_id)].constraints:
                        if c[0] == 'on top of':
                            surface_id = c[-1]
                            support_obj_id = c[-2]
                            support_obj_name = c[1]
                            surface_found = True
                    if not surface_found:
                        surface_id = 0
                        support_obj_id = pc_handler.floor_id
                        support_obj_name = 'floor'
                else:
                    surface_id = int(parent_id[-1])
                    support_obj_id = int(parent_id[0])
                    support_obj_name = pc_handler.objects[support_obj_id].obj_name

        elif 'surface' in reference_object_name:
            surface_id = int(objects_to_be_placed['root object']['id'].split('_')[-1])
            reference_object_name = objects_to_be_placed['root object']['grand_name']
            reference_object_id = objects_to_be_placed['root object']['grand_id']
            support_obj_id = int(reference_object_id)
            support_obj_name = reference_object_name
        else:
            surface_found = False
            skip = False
            try:
                f = int(reference_object_id)
            except:
                skip = True
            if not skip:
                for c in pc_handler.objects[int(reference_object_id)].constraints:
                    if c[0] == 'on top of':
                        if len(c) == 4:
                            surface_id = c[-1]
                            support_obj_id = c[-2]
                            support_obj_name = c[1]
                        else:
                            support_obj_id = c[-1]
                            support_obj_name = c[1]
                            s_surfaces = pc_handler.objects[support_obj_id].surfaces
                            support_surfaces_ele = [s['elevation'][-1] for s in s_surfaces]
                            
                            surface_id = 0 if len(support_surfaces_ele) <= 1 else np.argmin(np.abs(np.array(support_surfaces_ele)-pc_handler.objects[int(reference_object_id)].points[:, 2].min()))
                            
                        surface_found = True
            if not surface_found:
                surface_id = 0
                support_obj_id = pc_handler.floor_id
                support_obj_name = 'floor'

        # initlize_mesh(pc_handler, reference_object_name, int(reference_object_id), list_of_objects, inverse_rotation, inverse_translation, update_dummy=0)

        # rendered_info = pc_handler.render_object(render_only = [45], keep_other_objects = True, use_dummy_data=0, get_original_image=True)
        # rendered_info.save(os.path.join('/home/boudjoghra/projects/pc_pred/output/debug/init.jpg'))
        group_members = copy.deepcopy(list_of_objects)
        for m in group_members:
            m['object description'] = pc_handler.objects[int(m['id'])].description[0] if pc_handler.objects[int(m['id'])].description else pc_handler.objects[int(m['id'])].obj_name

        refinement_prompt = get_group_refinement_prompt(reference, floor_details, group_members, instruction, floor_notice=floor_notice)
        answer_refined = load_or_execute(root_path, f'instruction_in_queue_refined_{i}.yaml',load_prev, llm, refinement_prompt)
            
        list_of_objects_refined = parse_instruction_queue(answer_refined, valid_ids)
        list_of_objects_refined_id_to_details = {int(a['id']): a for a in list_of_objects_refined}
        list_of_objects_copy = copy.deepcopy(list_of_objects)
        for a in list_of_objects_copy:
            a['new_base_coordinate'] = list_of_objects_refined_id_to_details[int(a['id'])]['new_base_coordinate']
        list_of_objects_refined = list_of_objects_copy

        # initlize_mesh(pc_handler, reference_object_name, int(reference_object_id), list_of_objects_refined, inverse_rotation, inverse_translation, update_dummy=1)
        # rendered_info = pc_handler.render_object(render_only = [45], keep_other_objects = True, use_dummy_data=1, get_original_image=True)
        # rendered_info.save(os.path.join('/home/boudjoghra/projects/pc_pred/output/debug/init_refined.jpg'))
        # group_members = copy.deepcopy(list_of_objects_refined)
        # for m in group_members:
        #     m['object description'] = pc_handler.objects[int(m['id'])].description[0] if pc_handler.objects[int(m['id'])].description else pc_handler.objects[int(m['id'])].obj_name
        
        # orientation_refinement_prompt = get_group_orientation_refinement_prompt(reference, floor_details, group_members, instruction, floor_notice=floor_notice)
        # answer_orientation_refined = load_or_execute(root_path, f'instruction_in_queue_orientation_refined_{i}.yaml', reasoningllm, orientation_refinement_prompt)
        # list_of_objects_orientation_refined = parse_instruction_queue(answer_orientation_refined, valid_ids)
        list_of_objects_orientation_refined = list_of_objects_refined
        # initlize_mesh(pc_handler, reference_object_name, int(reference_object_id), list_of_objects_orientation_refined, inverse_rotation, inverse_translation, update_dummy=2)
        
        # rendered_info = pc_handler.render_object(render_only = [45], keep_other_objects = True, use_dummy_data=2, get_original_image=True)
        # rendered_info.save(os.path.join('/home/boudjoghra/projects/pc_pred/output/debug/init_refined_orient.jpg'))
        
        list_of_objects_orientation_refined_corrected = constraint_to_orientation_consistency(list_of_objects_orientation_refined)
        list_of_objects_orientation_refined_corrected = against_wall_distance(list_of_objects_orientation_refined_corrected, id_to_dim) if reference_object_name == 'wall' else list_of_objects_orientation_refined_corrected
        for obj_ in list_of_objects_orientation_refined_corrected:
            pc_handler.object_id_to_relative_details[int(obj_['id'])] = obj_
            pc_handler.object_id_to_relative_details[int(obj_['id'])]['inv_t'] = inverse_translation
            pc_handler.object_id_to_relative_details[int(obj_['id'])]['inv_r'] = inverse_rotation
            pc_handler.object_id_to_relative_details[int(obj_['id'])]['parent'] = f"{reference_object_name}/ID {reference_object_id}"
            if 'yes' in obj_['constraints']['facing']:
                facing_constraints[int(obj_['id'])] = int(reference_object_id)
        # list_of_objects_orientation_refined_corrected = fix_collision_case(list_of_objects_orientation_refined_corrected, pc_handler)
        initlize_mesh(pc_handler, reference_object_name, reference_object_id, list_of_objects_orientation_refined_corrected, inverse_rotation, inverse_translation, floor_elevation,support_obj_name=support_obj_name, support_obj_id=support_obj_id, surface_id=surface_id)
        # rendered_info = pc_handler.render_object(render_only = [45], keep_other_objects = True, get_original_image=True)
        # rendered_info.save(os.path.join('/home/boudjoghra/projects/pc_pred/output/debug/init_refined_orient_corr.jpg'))
        
        # graph_distances_prompt = get_distance_constraints_prompt((list_of_objects_orientation_refined_corrected, objects_to_be_placed), instruction, floor_xy_dims)
        # graph_distances_answer = load_or_execute(root_path, f'graph_distances_instruction_in_queue_orientation_refined_{i}.yaml', reasoningllm, graph_distances_prompt)
        # distances_graph = xmltodict.parse(graph_distances_answer.split('```')[1].replace('xml','').replace('\n',''))
        # list_of_objects_orientation_refined_corrected_optimized = get_optimized_coords(distances_graph, list_of_objects_orientation_refined_corrected)
        # initlize_mesh(pc_handler, reference['reference object details']['name'], list_of_objects_orientation_refined_corrected_optimized, inverse_rotation, inverse_translation)
        
    # o3d.io.write_triangle_mesh(os.path.join(root_path, 'llm_initialization.ply'), pc_handler.data_dummy[0])
    # o3d.io.write_triangle_mesh(os.path.join(root_path, 'llm_initialization_location_refined.ply'), pc_handler.data_dummy[1])
    # o3d.io.write_triangle_mesh(os.path.join(root_path, 'llm_initialization_loc_orientation_refined.ply'), pc_handler.data_dummy[2])
    # o3d.io.write_triangle_mesh(os.path.join(root_path, 'llm_initialization_loc_orientation_refined_corrected.ply'), pc_handler.data)
    # o3d.io.write_triangle_mesh(os.path.join(root_path, 'llm_initialization_loc_orientation_refined_corrected_inter_group_optimized.ply'), pc_handler.data)
    floor_group = None
    floor_group_id = None
    try:
        for g_i, g in enumerate(pc_handler.trainable_groups):
            if g[0] == pc_handler.floor_id:
                floor_group = g 
                floor_group_id = g_i
                break
        updated_groups = copy.deepcopy(pc_handler.trainable_groups)

        if floor_group_id is not None:
            for i in floor_group[1:]:
                for g_i, g in enumerate(pc_handler.trainable_groups):
                    if (g_i == floor_group_id):
                        continue
                    if i in g:
                        if i in updated_groups[g_i] and i in updated_groups[floor_group_id]: 
                            updated_groups[floor_group_id].remove(i)
            if len(updated_groups[floor_group_id]) == 1:
                updated_groups.remove(updated_groups[floor_group_id])
        updated_groups_ = copy.deepcopy(updated_groups)
        for g_i, g in enumerate(updated_groups):
            for g_mem in g:
                if (pc_handler.objects[g_mem].obj_name == 'wall') or not pc_handler.objects[g_mem].train:
                    updated_groups_[g_i].remove(g_mem)


        for g in updated_groups_:
            if len(g) == 1 and not pc_handler.objects[g[0]].train:
                updated_groups_.remove(g)
        # updated_groups = updated_groups_

        updated_groups = merge_intersecting_groups(updated_groups_)
        pc_handler.trainable_groups = updated_groups
        from functools import reduce
        def remove_duplicates(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]
        pc_handler.trainable_objects = reduce(lambda x, y: x + y, updated_groups_)
        pc_handler.trainable_objects = remove_duplicates([obj_id for obj_id in pc_handler.trainable_objects if pc_handler.objects[obj_id].train])
    
    except:
        pc_handler.trainable_groups = []
        pc_handler.trainable_objects = [obj.obj_id for obj in pc_handler.objects if obj.train]
    # for g in updated_groups:
    #     group_centers_surface_ids = []
    #     for g_item in g[1:]:
    #         for c in pc_handler.objects[g_item]:
    #             if 'on top of' == c[0]:
    #                 s_id = c[-1]
    #         group_centers_surface_ids.append((g[0]))
    edited_objects = ""
    for obj in [pc_handler.objects[idx] for idx in pc_handler.trainable_objects]:
        
        colors = obj.colors[:3]
        materials = obj.material[:3]
        pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
        description = obj.description[0] if obj.description else pred_name
        
        ref_obj = pc_handler.object_id_to_relative_details[obj.obj_id]['parent']
        if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
            pred_name = obj.obj_name
        obj_details = obj.get_obj_details(get_position_and_orientation=True)
        
        constraints = f'Facing object with ID {facing_constraints[obj.obj_id]}' if obj.obj_id in facing_constraints else "None"
        edited_objects += f"object id: {obj.obj_id}, class: {pred_name}, description: {description}, Relavite Coordinates to the {ref_obj} : ({pc_handler.object_id_to_relative_details[obj.obj_id]['new_base_coordinate']}), Orientation relative to the ({ref_obj}): ({pc_handler.object_id_to_relative_details[obj.obj_id]['new_orientation']}), object colors: {colors}, object material: {materials}, constraints:  {constraints}\n"
    pc_handler.previous_chat += f"""

    \n Previously edited objects from the instruction {instruction}:
        {edited_objects}
        It is important to note that the coordinates of the edited objects are in the coordinate frame of a reference object which is mentioned as well  
        While the other objects are in the absolute commun coordinate frame. 
    Make sure to use this information if the input instruction refers to correcting edited objects.
    """


def merge_intersecting_groups(groups):
    from collections import deque

    def find_and_merge(group, merged_indices):
        """Find all intersecting groups and merge them."""
        queue = deque([group])
        merged_set = set(group)
        root = group[0]  # Root is the first element of the first encountered group

        while queue:
            current = queue.popleft()
            for i, other in enumerate(groups):
                if i not in merged_indices and any(x in merged_set for x in other):
                    queue.append(other)
                    merged_set.update(other)
                    merged_indices.add(i)
                    
        merged_set.discard(root)  # Ensure root appears only once
        return [root] + sorted(merged_set)

    merged_indices = set()
    result = []
    
    for i, group in enumerate(groups):
        if i not in merged_indices:
            merged_indices.add(i)
            result.append(find_and_merge(group, merged_indices))

    return result

def initlize_mesh(pc_handler, root_object_name, root_obj_id, new_objects_locations, inverse_rotation, inverse_translation, floor_elevation, update_dummy=None, support_obj_name=None, support_obj_id = None, surface_id = None):
    # objects_in_transformed_frame = []
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    in_group_ids = [int(i['id']) for i in new_objects_locations]
    
    if update_dummy is None:
        if isinstance(root_obj_id, str) and 'group' not in root_obj_id:
            pc_handler.trainable_groups.append([int(root_obj_id)]+in_group_ids)
        else:
            pc_handler.trainable_groups.append(in_group_ids)
        # pc_handler.group_centers.append(root_obj_id)
    for new_locations in new_objects_locations:
        object_id = int(new_locations['id'])
        pc_handler.objects[object_id].constraints = []
        
        name = new_locations['name']
        new_base_coords = parse_and_compute(new_locations['new_base_coordinate'])
        new_orientation = float(parse_and_compute(new_locations['new_orientation'].split('=')[0])[0])
        # objects_in_transformed_frame = []
        # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        # object_in_new_frame = pc_handler.objects[int(object_id)].get_obj_details()
        # object_in_new_frame['base'] = new_base_coords
        # object_in_new_frame['orientation'] = int(new_orientation)
        # objects_in_transformed_frame.append(object_in_new_frame)

        # reference_obj_details_global_frame = pc_handler.objects[int(root_obj_id)].get_obj_details(get_position_and_orientation=True, get_surfaces=True)
        # transformed_color = [0.5,0,0]
        # original_color = [0,0.5,0]
        # added_in_new_frame_color = [0,0,0.5]
        # related_objects = pc_handler.get_graph_related_objects_in_ref_frame(root_obj_id, [], trasn_z_angle = -reference_obj_details_global_frame['oientation'], translate=-np.array(reference_obj_details_global_frame['base']))
        # reference_obj_details_no_surface = pc_handler.objects[int(root_obj_id)].get_obj_details(get_position_and_orientation=True, trasn_z_angle = -reference_obj_details_global_frame['oientation'], translate=-np.array(reference_obj_details_global_frame['base']))
        # trasnformed = visualize_bboxes([reference_obj_details_no_surface], get_geometrie = True, arrow_colors=transformed_color, shpere_colors=transformed_color)
        # reference_obj_details_original = pc_handler.objects[int(root_obj_id)].get_obj_details(get_position_and_orientation=True)
        # related_objects_original = pc_handler.get_graph_related_objects_in_ref_frame(root_obj_id, [])
        # original = visualize_bboxes([reference_obj_details_original], get_geometrie = True, arrow_colors=original_color, shpere_colors=original_color)
        # added_in_new_frame = visualize_bboxes(objects_in_transformed_frame, get_geometrie = True, arrow_colors=added_in_new_frame_color, shpere_colors=added_in_new_frame_color)
        # o3d.visualization.draw_geometries([coordinate_frame]+added_in_new_frame+original+trasnformed)
        old_info = pc_handler.objects[object_id].get_obj_details(get_position_and_orientation=True, get_surfaces=True)
        t = inverse_rotation@np.array(new_base_coords)+inverse_translation-old_info['base']
        rel_rot_angle = np.radians(new_orientation-old_info['oientation'])
        # old_orien = np.radians(old_info['oientation'])
        # new_orien = np.radians(rel_rot_angle)

        rotation_matrix = np.array([
                            [np.cos(rel_rot_angle), -np.sin(rel_rot_angle), 0],
                            [np.sin(rel_rot_angle),  np.cos(rel_rot_angle), 0],
                            [0,                 0,                  1]
                        ])@inverse_rotation
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = t

        r_obj_id = int(root_obj_id.split('_')[0]) if isinstance(root_obj_id, str) else int(root_obj_id)
        if floor_elevation == new_base_coords[-1]:
            support_obj_id = pc_handler.floor_id
            surface_id = 0
        else:
            if support_obj_id != root_obj_id and root_object_name != 'wall':
                pc_handler.maintain_distance[object_id] = (root_obj_id, np.sqrt(new_base_coords[0]**2+new_base_coords[0]**2))
            if support_obj_id == 'floor':
                support_obj_id = pc_handler.floor_id
                surface_id = 0
        if len(pc_handler.objects[support_obj_id].surfaces) != 0:
            support_elevation = pc_handler.objects[support_obj_id].surfaces[surface_id]['elevation'][-1].item()
        else:
            support_elevation = pc_handler.objects[support_obj_id].points[:,2].max()
        transformation_matrix[2, -1] = support_elevation-pc_handler.objects[object_id].points[:,2].min()
        transformed_group = pc_handler.get_linked_groups()[object_id]+[object_id]
        for g_ele_i in transformed_group:
            pc_handler.update_point_cloud(g_ele_i, transformation_matrix, object_id, update_dummy=update_dummy)
        
        if update_dummy is None:
            
            pc_handler.objects[object_id].train = True if np.linalg.norm(t) > 0.05 else False
            # pc_handler.objects[object_id].constraints = [['maintain', pc_handler.objects[i].obj_name, i] for i in in_group_ids]

            # if (root_object_name != 'floor') and ('none' not in new_locations['constraints']['in_surface'].lower()):
            pc_handler.trainable_object_to_support[object_id] = support_obj_id
            pc_handler.objects[object_id].constraints.append(['on top of', support_obj_name, support_obj_id, int(surface_id)])
            if root_object_name == 'wall': 
                pc_handler.objects[object_id].constraints.append(['against', 'wall', root_obj_id])
                # pc_handler.objects[object_id].constraints.append(['maintain', root_object_name, root_obj_id])
            