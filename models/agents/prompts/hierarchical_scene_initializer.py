import numpy as np
import copy
import numpy as np
from scipy.spatial import distance

def group_objects_by_proximity(objects, threshold=1.0):
    groups = []
    
    for obj in objects:
        placed = False
        for group in groups:
            if any(distance.euclidean(obj['base'][:2], g['base'][:2]) < threshold for g in group):
                group.append(obj)
                placed = True
                break
        if not placed:
            groups.append([obj])
    
    return groups
def compute_region_bounds(group):
    if len(group) == 1:
        return group[0]['dimensions'][0], group[0]['dimensions'][1], (group[0]['base'][0], group[0]['base'][1])
    
    x_mins = [obj['base'][0] - obj['dimensions'][0] / 2 for obj in group]
    x_maxs = [obj['base'][0] + obj['dimensions'][0] / 2 for obj in group]
    y_mins = [obj['base'][1] - obj['dimensions'][1] / 2 for obj in group]
    y_maxs = [obj['base'][1] + obj['dimensions'][1] / 2 for obj in group]
    
    dx = max(x_maxs) - min(x_mins)
    dy = max(y_maxs) - min(y_mins)
    center_x = np.mean([min(x_mins), max(x_maxs)])
    center_y = np.mean([min(y_mins), max(y_maxs)])
    
    return dx, dy, (center_x, center_y)

def generate_compact_description(objects, threshold=1.):
    grouped_objects = group_objects_by_proximity(objects, threshold=threshold)
    descriptions = []
    
    for group in grouped_objects:
        names = list(set(obj['name'] for obj in group))
        dx, dy, (cx, cy) = compute_region_bounds(group)
        description = f"{', '.join(names)} are located in region dx = {dx:.2f}, dy = {dy:.2f}, centered at x = {cx:.2f}, y = {cy:.2f}).\n"
        descriptions.append(description)
    
    return descriptions

def get_prompt(pc_handler, objects_placement, instruction, floor_notice, exclude_ids):

    root_object = objects_placement['root object']
    children = objects_placement['children']
    for c in children:
        obj_details = pc_handler.objects[int(c['id'])].get_obj_details(get_position_and_orientation=True)
        dimensions = pc_handler.objects[int(c['id'])].get_obj_details(get_position_and_orientation=True, trasn_z_angle = -obj_details['oientation'])['dimensions']
        c['dimensions'] = dimensions
    
    root_object_name_llm = root_object['name']
    root_object_id_llm = root_object['id']
    if 'floor' not in root_object_name_llm and root_object_name_llm in pc_handler.class_names:
        root_object_name = pc_handler.objects[int(root_object_id_llm)].obj_name
        if 'cc' not in root_object_name and root_object_name_llm != root_object_name:
            root_object['id'] = str(pc_handler.class_names.index(root_object_name_llm))
    surface_id = None
    if isinstance(root_object['id'], str) and 'group'in root_object['id']:
        reference_obj_id = pc_handler.floor_id
    elif root_object['name'] != 'floor':
        if 'surface' not in root_object['name'].lower():
            reference_obj_id = root_object['id']
        else:
            reference_obj_id = int(root_object['id'].split('_')[0])
            surface_id = int(root_object['id'].split('_')[1])
    else:
        reference_obj_id = pc_handler.floor_id
    
    reference_obj_details_global_frame = pc_handler.objects[int(reference_obj_id)].get_obj_details(get_position_and_orientation=True, get_surfaces=True)
    if 'group'in root_object['id']:
        del reference_obj_details_global_frame['dimensions']
    if surface_id is None:
        surface_elevation = 0 
    elif pc_handler.objects[int(reference_obj_id)].surfaces:
        surface_elevation = pc_handler.objects[int(reference_obj_id)].surfaces[surface_id]['elevation'][-1].item()
    else:
        surface_elevation = pc_handler.objects[int(reference_obj_id)].points[:,-1].max()

    object_translation = np.array(reference_obj_details_global_frame['base'])
    object_translation[-1] = surface_elevation if surface_elevation != 0 else object_translation[-1] 
    reference_obj_details = pc_handler.objects[int(reference_obj_id)].get_obj_details(get_position_and_orientation=True,get_surfaces=True, trasn_z_angle = -reference_obj_details_global_frame['oientation'], translate=-object_translation)
    if 'group'in root_object['id']:
        # reference_obj_details['dimensions'] = ['unknown', 'unknown', 'unknown']
        reference_obj_details['id'] = root_object['id']
        reference_obj_details['name'] = root_object['name']
        del reference_obj_details['surfaces']
        del reference_obj_details['dimensions']
    inverse_transform = {'angle_z': reference_obj_details_global_frame['oientation'], 'translation': object_translation}

    if 'group' not in root_object['id']:
        related_objects = pc_handler.get_graph_related_objects_in_ref_frame(reference_obj_id, exclude_ids, trasn_z_angle = -reference_obj_details_global_frame['oientation'], translate=-object_translation)
        related_objects_raw = copy.deepcopy(related_objects)
        relevent_classes = []
        if surface_id is not None:
            dimensions = reference_obj_details['dimensions']
            dimensions[-1] = 0.002
            proximity_distance = ((dimensions[0]**2+dimensions[1]**2)**0.5)/4
            dimensions[-1] = 0.02
            base_coords = reference_obj_details['base']
            base_coords[-1] = 0
            reference_obj_details = {f"{reference_obj_details['name']}'s id": reference_obj_details['id'], 'name': f'Surface ID {surface_id} of the '+ reference_obj_details['name']+f" with ID {reference_obj_details['id']}", 'Surface dimensions': dimensions, 'Surface base': base_coords}
            related_objects_value = 'Surface is empty'
            for k, v in related_objects.items():
                if 'ID '+str(surface_id) in k:
                    related_objects_value = v
                    
                    break
            related_objects = {}
            if related_objects_value != 'Surface is empty':
                relevent_classes.extend([obj['name'] for obj in related_objects_value])
            related_objects[f"Other objects in the surface ID {surface_id}"] = generate_compact_description(related_objects_value, proximity_distance) if isinstance(related_objects_value, list) else related_objects_value
    else:
        related_objects = 'None'
        relevent_classes = []
        
    reference_obj_details_llm_in = copy.deepcopy(reference_obj_details)

    key_name_dim = 'dimensions' if surface_id is None else 'Surface dimensions'
    key_name_base = 'base' if surface_id is None else 'Surface base'
    if key_name_dim in reference_obj_details_llm_in.keys():
        reference_obj_details_llm_in[key_name_dim] = f"""dx = {reference_obj_details_llm_in[key_name_dim][0]}, dy = {reference_obj_details_llm_in[key_name_dim][1]}, dz = {reference_obj_details_llm_in[key_name_dim][2]}"""
    reference_obj_details_llm_in[key_name_base] = f"""x = {reference_obj_details_llm_in[key_name_base][0]}, y = {reference_obj_details_llm_in[key_name_base][1]}, z = {reference_obj_details_llm_in[key_name_base][2]}"""
    # reference_obj_details_llm_in
    if (reference_obj_details_llm_in['name'] == 'wall') and ('surfaces' in reference_obj_details_llm_in.keys()):
        del reference_obj_details_llm_in['surfaces']
    if surface_id is None:
        reference_obj_details_llm_in['orientation'] = reference_obj_details_llm_in.pop('oientation')

    reference = {'reference object details': reference_obj_details_llm_in, 'related objects to reference': related_objects}
    reference_txt = 'reference object details: '+ str(reference_obj_details_llm_in)+ '\n' +'related objects: ' + str(related_objects)
    floor_info = pc_handler.objects[pc_handler.floor_id].get_obj_details(get_surfaces=True, trasn_z_angle = -reference_obj_details_global_frame['oientation'], translate=-object_translation)
    floor_elevation = float(floor_info['surfaces'][0]['elevation'].replace('meters', ''))
    if (root_object['name'] != 'floor') and (surface_id is None):
        floor_details = floor_info
        floor_details = f"""
        USe the floor elevation to figure out where to place objects that should be on the floor
        Floor details: {floor_details}
        """
    else:
        floor_details = ''
    
    obj_details = pc_handler.objects[pc_handler.floor_id].get_obj_details(get_position_and_orientation=True)
    floor_x_y_dimensions = pc_handler.objects[pc_handler.floor_id].get_obj_details(get_position_and_orientation=True, trasn_z_angle = -obj_details['oientation'])['dimensions']
    floor_x_y_dimensions = [floor_x_y_dimensions[0], floor_x_y_dimensions[1]]

    objects_to_be_placed = '\n'
    for c in children:
        objects_to_be_placed += f"Object ID: {c['id']}, Object Name: {c['name']}, object dimensions: [dx = {c['dimensions'][0]}, dy = {c['dimensions'][1]}, dz = {c['dimensions'][2]}], Instruction: {c['instruction']}\n"
    relevent_classes = list(set(relevent_classes))
    relevent_classes = [c for c in relevent_classes if 'cc' not in c]
    prompt = f"""
    You will be given a reference object and a list of objects that you need to place relative to a reference object. Your role is to think step by step and seggest new locations, orientations, and constraints for the list of objects.
    Each object has to be placed following its instruction, you must suggest 3D coordinate for the base of the object, an orientation of the object, and a list of constraints with respect to the reference object.

    The representation of the reference object:
        - The reference object is represented with 
            1. Its base coordinate, which represents the 3D coordinate of the object with the minimum elevation (z) in meters and center in x and y.
            2. Its dimensions which represent the height(following the z axis), the width(following the x axis), the depth(following the y axis) in meters
            3. Its orientation, which refers to the orientation of the object around the z axis, in degrees.
            4. Its surfaces which can be used for placing objects, each surface has an ID where id 0 represents the surface with the highest elevation, the elevation is in meters.
            5. List of objects that are on top of the object

    The representation of the List of object to be placed relative to the reference object {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id}:
            1. its base coordinate, which represents the 3D coordinate of the object with the minimum elevation (z) in meters and center in x=0 and y=0.
            2. Its orientation, which refers to the orientation of the object around the z axis, in degrees.

    The possible list of constraints with respect to the reference object {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id} are: 
        - in_surface : this constraints concerns only instructions that require placing objects inside or on top of a reference object, if the instruction requires placing an object on top, that means the surface ID is 0 since it is the one with the highest elevation
        - facing: This constraint concerns objects which should be facing the reference object in a natural setting, for example a chair should be facing a referebce object whiteboaard.


    Important details when reporting the final location and orientation:
        - Don't perform the math operation instead report the formula with values and operations to get the new location or orientation. The allowed operations are :*: for multiplication, /: for division, -: for minus, +: for summation, cos: for cosine function,  sin: for sin function. angles should be in degrees.

    Hint on relations between objects:
        - If an object is facing another, its orientation should be 180 degrees minus the orientation of the other object
        - if an object is facing the same direction as another, both should have the same orientation

    Important details on the reference object's {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id} orientation and location:
        - the reference object is oriented towards (meaning its front) the positive direction of the x-axis. And its base coordinate is located near the origin.


    Output format:
        Please end your step by step thinking with the following xml which summarizes the new locations , new orientations and list of constraints of objects with respect to the reference object relative to the reference object {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id}:
        Important: place only final result in between tags, don't include the operator = 
    xml ```
    <objects>
        <object>
            <id>[please place the object ID here]<id>
            <name>[please place the object name here]<name>
            <new_base_coordinate>[please place here the new base coordinate of the object]</new_base_coordinate>
            <new_orientation>[please place here the new orientation the object]</new_orientation>
            <constraints>
                <facing>[please mention here wheather the object should be facing the reference object {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id} or not, answer with yes or no]<facing>
                <facing_id>[Please put here the ID of the object that should be faced]</facing_id>
                <in_surface>[If the object should be in one of the surfaces place the surface ID here, if not place None]<in_surface>
                <distanace_to_reference>[please place the distance to the reference object here]<distanace_to_reference>
            </constraints>
        <object>
    <objects>

    ```  

    Reference Object: 
    {reference_txt}

    List of Objects to be placed with their instructions: 
    {objects_to_be_placed} 

    {floor_details}

    Now please proceed with your suggested new coordinates and orientations, taking into account that the reference object x center and y center are both 0,0 and this is similar to its base coordinate. It is also important to note that all objects are represented with base coordinate which is the center of the object in x and y and minimum elevation in z.

    Important: the front of the reference object  is in the positive x axis, thus if an object should be in front of it it must be placed at x>0 and y that is bounded.
    Important: you must use place the objects elevation (z axis) at the floor level if the object should be on the floor.
    Important regarding in_surface constraint: if the reference object is a surface of an object, please put the ID of the surface (which is a number, don't put something like fridge's surface or table's surface) in this constraint.

    Very important: Pay attention to the instruction for each object, if an object should not be moved please don't include it in the xml list. If no object should be moved return an empty xml <objects></objects>


    Left, right, front, and back sides conventions:
        Left side with respect to the reference object is : y < 0
        Right side with respect to the reference object is : y > 0
        Front side with respect to the reference object is : x > 0
        Back side with respect to the reference object is : x < 0

    Very important when placing objects against walls or other objects:
        If an object is placed against a wall or another object <facing> should be no and the orientation of the object to be place should be the same as the reference object 
        If an object is against the wall, its x should be half the dx dimension of the object and y can range from -dy_wall/2 to dy_wall/2

    Very important for the <facing> constraint which are with respect to the reference object {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id}: 
        If an object should be facing the same direction as the reference object {pc_handler.objects[int(reference_obj_id)].obj_name} id {reference_obj_id} <facing> constraint should be set to no 
        Example (helpful for reasoning only):
            - place chairs with respect to another reference object chair for a role that requires them to be in the same direction like guest watching TV, means that <facing> is no
            - place chairs with respect to another reference object TV for a role that requires them to be in the same direction like guest watching TV, means that <facing> is yes 
    """
    return prompt, inverse_transform, reference, floor_details, objects_placement, floor_x_y_dimensions, floor_elevation

def get_group_refinement_prompt(reference, floor_details, group_members, instruction, floor_notice):
    prompt = f"""
    You are a helpful assistant who will get several group members and a reference object. Your role is to analyze the puprose of the group and refine the object details in the input.

    Details on the group:
        - The group of object that you must analyze is created by a designer following an instruction. This designer might make mistakes in correctly executing the instruction, giving wrong constraints, or provide unrealistic 
        distances between group members given the functionality of the group. Or wrong constraints with respect to the reference object.

    What you should maintain in the input:
        - keep the high level structure between group members the same. for example if you notice a grid structure or a certain pattern between object you must move object while maintain this structure.
    

    Details on the object format:
        An object is represented by:
            1. Its base coordinate, which represents the 3D coordinate of the object with the minimum elevation (z) in meters and center in x and y.
            2. Its dimensions which represent the height(following the z axis), the width(following the x axis), the depth(following the y axis) in meters
            3. Its orientation, which refers to the orientation of the object around the z axis, in degrees.
            4. A small description of the object
        
        For each group member, a list of constraints is provided with respect to the reference object:
        The possible list of constraints with respect to the reference objectare: 
        - in_surface : this constraints concerns only instructions that require placing objects inside or on top of a reference object, if the instruction requires placing an object on top, that means the surface ID is 0 since it is the one with the highest elevation
        - facing: This constraint concerns objects which should be facing the reference object in a natural setting, for example a chair should be facing a referebce object whiteboaard.

        Note: distances are not measured from center to center, but from object surface to surface. For example, if two chairs are touching their distance is 0 meters. 
    
    Output format:
        Please end your step by step thinking with the following xml which summarizes the new locations , new orientations and list of constraints of objects with respect to the reference object:
    xml ```
    <objects>
        <object>
            <id>[please place the object ID here]<id>
            <name>[please place the object name here]<name>
            <new_base_coordinate>[please place here the new base coordinate of the object]</new_base_coordinate>
            <new_orientation>[please place here the new orientation the object]</new_orientation>
            <constraints>
                <facing>[please mention here wheather the object should be facing the reference object or not, answer with yes or no]<facing>
                <in_surface>[If the object should be in one of the surfaces of the the reference object, please place the surface ID here, if not place None]<in_surface>
                <distanace_to_reference>[please place the distance to the the reference object here]<distanace_to_reference>
            </constraints>
        <object>
    <objects>

    ```  
    The instruction that was given to the designer is:
    {instruction}

    Reference object:
    {reference}

    Group members:
    {group_members}

    {floor_details}

    What you should refine:
        - Please analyse the distances between group members if its too small or too large, make sure that the arrangement is realistic and common.
        - Please analyse the orientation of objects relative the reference if you find anything unusual please fix it. For example if an object should face the reference object but the current location and orientation doesn't reflect that.
        - Analyze the constraints which are relative to the reference object, then update them.
    

    What you should preserve in the input:
        - The high level structure of the group members. Try to spot patterns in the layout and keep it.
        - The relative position of objects with respect to each other, please maintain the relative position of objects relative to each other.
        - The orientation of the objects
    
    Hint: common ditance between the closest two objects of the same group members is generally less then 0.5 meters. Unless it is a memeber like whiteboard or TV which can be functional from far. Please keep this in mind unless provided in the instruction.

    Important note:
        - Take into account that the reference object x center and y center are both 0,0 and this is similar to its base coordinate. It is also important to note that all objects are represented with base coordinate which is the center of the object in x and y and minimum elevation in z. After deciding on the location use the orientation codebook to adjust the orientation of the object if the object must face the reference object.
    
    
    Very Important: the front of the reference object is in the positive x axis, thus if an object should be in front of it it must be placed at x>0 and y that is bounded.
    Important regarding in_surface constraint: if the reference object is a surface of an object, please put the ID of the surface (which is a number, don't put something like fridge's surface or table's surface) in this constraint.
    Important Tip about moving or orienting objects: certain objects are best to stay where they are, for example requesting to place a chair to watch TV means that the TV is best to state in its location. So please don't move objects unless necessary.

    Important: The instruction is already complete; you just need to refine the pairwise x,y distances between objects to ensure they are realistic. If an object should naturally be on the floor (for example a chair, armchair, table, cabinet) but its z-elevation indicates otherwise, and the instruction does not specify a different surface, adjust its zz-elevation to match the floor level.

    Very important when placing objects against walls or other objects:
        If an object is placed against a wall or another object <facing> should be false and the orientation of the object to be place should be the same as the reference object 
        If an object is against the wall, its x should be half the dx dimension of the object and y can range from -dy_wall/2 to dy_wall/2

    Extremely important: Yous must return all {len(group_members)} objects with their adjusted distances, don't forget any object.
    """
    return prompt