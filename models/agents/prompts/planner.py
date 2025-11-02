def get_prompt(pc_handler, instruction = '', placement_plan=None, dependency_plan=None, use_gt=False, relevent_ids=None, relevent_classes=None, type = 'relevent classes', relevent_relations_ids=None, editing_type='rearrange'):

    if type == 'placement plan':
        objects_details = ""
        retrieved_ids = [obj.obj_id for obj in relevent_ids]
        node_relations = []
        exclude_from_main_graph = []
        Surface_Note = ""
        
        for node in relevent_relations_ids:
            exclude_from_main_graph.append(int(node['id']))

            related_objects_dict = pc_handler.get_graph_related_objects_in_ref_frame(int(node['id']))
            obj = pc_handler.objects[int(node['id'])]
            colors = obj.colors[:3]
            materials = obj.material[:3]
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
            if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                pred_name = obj.obj_name
            obj_details = obj.get_obj_details(get_position_and_orientation=True)

            if use_gt and ('cc' in obj.obj_name or obj.obj_name == 'unknown'):
                continue

            objects_details += f"object id: {obj.obj_id}, class: {pred_name}, Dimensions: (dx = {obj_details['dimensions'][0]}, dy = {obj_details['dimensions'][1]}, dy = {obj_details['dimensions'][2]}),  Coordinates: (x = {obj_details['base'][0]}, y = {obj_details['base'][1]}), object colors: {colors}, object material: {materials} \n"
            objects_details += pc_handler.get_max_hights_for_surfaces(int(node['id']))+'\n'
            # close_to = 'close' in node['relation'] if node['relation'] is not None else False
            close_to = False
            on_top = 'on top' in node['relation'] if node['relation'] is not None else False
            surface_ids = [int(id) for id in node['surface_ids'].split(',')] if on_top else []
            for edge, related_nodes in related_objects_dict.items():

                related_nodes_text = ''
                for r_node in related_nodes:
                    exclude_from_main_graph.append(int(r_node['id']))

                    related_nodes_text += f"A {r_node['name']} with id = {r_node['id']} and dimensions (dx = {r_node['dimensions'][0]}, dy = {r_node['dimensions'][1]}, dz = {r_node['dimensions'][2]})"
                if close_to and 'close' in edge:
                    objects_details += 'Objects to relevent to the instruction: \n'
                    
                    objects_details += f"The {pred_name}, with id {obj.obj_id} has \n"
                    objects_details += f"           {edge} : {related_nodes_text} \n"
                if on_top and any([str(f'surface ID {s_id}') in edge for s_id in surface_ids]):
                    objects_details += 'Objects to relevent to the instruction: \n'
                    
                    objects_details += f"The {pred_name}, with id {obj.obj_id} has \n"
                    objects_details += f"           {edge} : {related_nodes_text} \n"
                    Surface_Note = f"""Important regarding surface IDs: A surface is the area where objects can be placed in an object, each object has surfaces with several IDs from 0 and up, the surface with the lowest elevation has the highest ID.
                                        Example for surfaces: if there are three surfaces with ids 0,1,2,3 the lowest id corresponds to the top surface in this case ID 0 where the highest ID corresponded to the lowest surface in this case 3."""
        for obj in relevent_ids:
            if obj.obj_id in exclude_from_main_graph:
                continue
            colors = obj.colors[:3]
            materials = obj.material[:3]
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
            if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                pred_name = obj.obj_name
            obj_details = obj.get_obj_details(get_position_and_orientation=True)


            objects_details += f"object id: {obj.obj_id}, class: {pred_name}, Dimensions: (dx = {obj_details['dimensions'][0]}, dy = {obj_details['dimensions'][1]}, dy = {obj_details['dimensions'][2]}), Coordinates: (x = {obj_details['base'][0]}, y = {obj_details['base'][1]}), object colors: {colors}, object material: {materials} \n"

        prompt = f"""
            You will be given a list of objects with their ids colors and materials, you have to suggest a plan on how to place objects to execute the instruction.

            Important analysis before planning how objects should be moved:
                Analyze the instruction and keep the movement of objects to the minimum, while insuring the instruction is satisfied. For example if the goal is to create a seating area to whatch TV, the TV should not be moved.

                
            Objects list with their object ids, class name, list of major colors, list of materials:

            {objects_details}

            {pc_handler.previous_chat}

            Instruction is :
            {instruction}

            Important: enphasis on objects that should not be moved in the plan, an example is " place the chairs to watch TV" the TV should remain untouched and the chairs should be placed in front of it while facing it.
            Please end your thinking with <placement_plan>[place your detailed plan by specifying object ids and class names here]</placement_plan>
            Important: enphasis on objects that should not be moved in the plan, an example is " place the chairs to watch TV" "arrange chairs for presentation on a screen" the TV and screen should remain untouched and the chairs should be placed in front of it while facing it.
            {Surface_Note}

            Important when suggesting the plan: 
                - please suggest a target location for every object you want to move relative to other objects or the floor, even if the instruction is vague. 
                    For example if an instruction says to clear a table, you must identify what objects are on the table and suggest new locations for these objects 
                - First analyze all potential target location, and place object relative to the most logical targets. Example, an instruction 'move the chair' has multiple targets but the most logical one is close to a table. 
                - If the target location is support object, please make sure to suggest which surface among the object surfaces it should go to in the plan
                - Some objects are better stacked on each other, for example papers and boxes shoulkd be stacked on each other if they are to be organized. where the largest one with dx,dy is the first.
                - The object level instruction must refer to an object (if desired to be placed close to it) or one of its surfaces (if desired to be placed on a surface) that exists in the list of objects.
                - Make the plan with natural language only, don't suggest to place objects in specific coordinates.
                - It is very important to pay attension to objects coordinates, since there are multiple objects with same functionality but different sizes. e.g. a large plant cannot be placed on top of a table, but a midsized or small plant one can be which gives a better vibe to the space.
                
            When should you stack objects on top of each other:
                - Some objects like papers or boxes are better stacked on top of each other, in this case the plan should be:
                    - place largest box on some surface 0 of another object or floor.
                    - place second smallest box on the placed box.
                    - and so on ... 
                - If three objects a (largest),b(smallest),c(medium) are to be stacked on top of each other a should be on a surface floor, or table, etc as it is the base of the stack, c should be on surface ID 0 of a, and b should be on surface ID 0 of c.   

            When to use the coordinates:
                - Each objects has x,y coordinates use them to figure out far or close objects if mentioned in the instruction. It is strict to not include any coorcinates in the final placement plan.

            How to handle instructions with few details:
                - Try to make a plan that can be physically plausible, for example placing 10 sofas for a presentation can not be done in one row, you have to place them in multiple rows in this case 3 rows would be good. 

    
        
          """ 

    if type == 'generate group':
        objects_details = ""
        # retrieved_ids = [obj.obj_id for obj in relevent_ids]
        # node_relations = []
        exclude_from_main_graph = []
        Surface_Note = ""
        for node in relevent_relations_ids:
            exclude_from_main_graph.append(int(node['id']))

            related_objects_dict = pc_handler.get_graph_related_objects_in_ref_frame(int(node['id']))
            for edge, related_nodes in related_objects_dict.items():
                related_nodes_text = ''
                for r_node in related_nodes:
                    exclude_from_main_graph.append(int(r_node['id']))

                    related_nodes_text += f"A {r_node['name']} with id = {r_node['id']} and dimensions (dx, dy, dz) = {r_node['dimensions']}"
                if node['relation'] is None:
                    continue
                if any([r in edge for r in node['relation'].split(',')]):
                    obj = pc_handler.objects[int(node['id'])]
                    colors = obj.colors[:3]
                    materials = obj.material[:3]
                    pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
                    if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                        pred_name = obj.obj_name
                    objects_details += f"object id: {obj.obj_id}, class: {pred_name}, object colors: {colors}, object material: {materials} \n"
                    objects_details += f"           {edge} : {related_nodes_text} \n"
                    Surface_Note = f"""Important regarding surface IDs: A surface is the area where objects can be placed in an object, each object has surfaces with several IDs from 0 and up, the surface with the lowest elevation has the highest ID.
                                        Example for surfaces: if there are three surfaces with ids 0,1,2,3 the lowest id corresponds to the top surface in this case ID 0 where the highest ID corresponded to the lowest surface in this case 3."""


        for obj in relevent_ids:
            if obj.obj_id in exclude_from_main_graph:
                continue
            colors = obj.colors[:3]
            materials = obj.material[:3]
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
            if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                pred_name = obj.obj_name
            objects_details += f"object id: {obj.obj_id}, class: {pred_name}, object colors: {colors}, object material: {materials} \n"
        # Objects list with their object ids, class name, list of major colors, list of materials:
        # {objects_details}

        prompt = f"""
        You will be given a list of objects with their ids colors and materials and a plan to place these object. Your role is to return the logical dependency between objects and format the placement of objects in a hierarchical manner starting from the floor. Please strictly follow the placement plan

        Important analysis before planning how objects should be moved:
            Analyze the instruction and keep the movement of objects to the minimum, while insuring the instruction is satisfied, pay attention to the placement plan, some objects are best to be kept unchouched you need to reflect that in the object level instruction in the hierarchy. For example if the goal is to create a seating area to whatch TV, the TV should not be moved. In this case the TV will nest the objects for seating, while having the instruction "keep the TV untouched".


        Placement plan: 'place the table with id 10 near the door id 12 and a bottle with id 1 on top of the table (surface ID 0 of the table) with id 10, and the chair id 50 facing the table.'

        Hierarchical Structure for Object Placement

            The hierarchy follows a nested dependency model, where objects are placed relative to their parent objects. This ensures spatial constraints are logically maintained.
            1. Root Level (Floor)

                The floor is the base of the environment, meaning all objects are ultimately placed on it.
                The floor itself remains static and untouched, serving as the foundational layer for all placements.

            2. First Nested Level (Door)

                The door (ID 12) is placed directly on the floor, meaning it is positioned independently.
                Since the door is static like the floor, it does not move or act as a container for other objects.

            3. Second Nested Level (Table)

                The table (ID 10) is placed near the door (ID 12).
                This means the table's position is spatially related to the door but not contained within it.
                Since the table is a movable object, its placement depends on the door's position.

            4. Third Nested Level (Chair & Surface)

                The chair (ID 50) is placed facing the table (ID 10), making it dependent on the table for its orientation.
                The table’s surface (ID 0) is an implicit subcomponent of the table and serves as a placement area for smaller objects.
                While the surface is not a separate object, it acts as a reference point for placing items on the table.

            5. Fourth Nested Level (Bottle)

                The bottle (ID 1) is placed on top of the table (specifically, surface ID 0).
                Since the surface belongs to the table, the bottle is indirectly dependent on the table’s placement.

            Purpose of the Hierarchy

            The structure enforces a logical dependency between objects. For example:
                The bottle’s placement depends on the table.
                The table’s placement depends on the door.
                The chair and surface placement depends on the table.
                The door’s placement depends on the floor.

            This hierarchical representation reflects real-world relationships and ensures clarity in placement instructions.


        How to approach this:
        - You need to figure out the group center and what objects but be placed relative to the group centers.
        - The group centers are placed relative to the floor, while the group memebers are placed relative to the group centers.
        - If an object is placed in relation to other object it should be nested in it.


        Final format (You must follow this output format):
        xml ```
            <object>
                <name>floor</name>
                <id>floor_id</id>
                <instruction>leave the floor untouched as floors cannot be moved</instruction>
                <object>
                    <name>door</name>
                    <id>12</id>
                    <instruction>leave the door untouched</instruction>
                    <object>
                        <name>table</name>
                        <id>10</id>
                        <instruction>place the table close to the door with id 12</instruction>
                        <object>
                            <name>surface</name>
                            <id>0</id>
                            <instruction>Leave surface ID 0 untouched as it is part of the table</instruction>
                                <object>
                                    <name>bottle</name>
                                    <id>1</id>
                                    <instruction>place the bottle on top of the surface with ID 0</instruction>
                                </object>
                        </object>
                        <object>
                            <name>chair</name>
                            <id>50</id>
                            <instruction>place the chair id 50 in front of the table, facing the table</instruction>
                        
                        </object>

                    </object>
                </object>
            </object>
        ```

            
        Extremely Important for the nesting:
            Even if an object doesn't move it must nest related objects.

        Now please proceed with the following:


        Placement plan :
        {placement_plan}

        Denependency plan:
        {dependency_plan}

        Structured xml hierarchy:
        Please proceed with the step by step thinking then end it with xml output here
        Important notes: 
            - If an object is placed in relation to other object (on top of, close to, near etc) it should be directly nested in it.
            - Please nest the desired surface in its correspoding object parent if exists in the placement plan.
            - An object is nested only if it should remain untouched or placed relative to the parent, but not moved from the parent.
                        Example when not to nest: the cup should not be nested in the fridge in 'move the cup from the fridge' 
                        Example when to nest: the cup should be nested in surface id 1 and surface id 1 should be nested in the fridge in 'place the cup in surface 1 in the fridge' 

        Important regarding object level instructions:
            - Add the word 'untouched' in the instruction if the object is a surface or should not be moved. Surface instruction should always be 'leave surface untouched'

        Very important: 
            - You must return the structure that enforces a logical dependency (use the dependancy plan in order to figure out which object is nested in which) between objects to figure out which object relates to which object before generating the xml nests
            - If an object is not explecitly mentioned to be moved in the placement plan you should leave it untouched, you your role is to structure the placement plan in a nested structure following the dependency plan


            Extremely important note: You must move only the objects that are mentioned to be moved in the instruction. For example 'move glass to the fridge' means the fridge must stay in place 
        """ 
    return prompt

