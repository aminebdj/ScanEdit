def get_prompt(pc_handler, instruction = '', placement_plan=None, dependency_plan=None, use_gt=False, relevent_ids=None, relevent_classes=None, type = 'relevent classes', relevent_relations_ids=None, editing_type='rearrange'):
    if type == 'relevent classes':
        promptable_classes = []

        for obj in pc_handler.objects:
            if obj.obj_name == 'corrupt':
                continue
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
            
            if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                pred_name = obj.obj_name
            promptable_classes.append((pred_name, obj.obj_id))
        prompt = f"""
        You will be given a list of class names and your role is located the classes of interest. The classes of interest are objects to be moved and potential target locations. The classes don't have to be explecitely mentioned in the prompt, you must infer them from the context. especially the target locations where objects should be moved.  
        Among the following classes <objects>{list(set([o[0] for o in promptable_classes]))}</objects> which ones are relevent to "{instruction}". 
        PLease give only the necessary class names to execute the task and make sure to return them in the tag <objects>[place list of objects here]</objects>.
        Tips: 
            - Relevant classes do not necessarily need to be explicitly mentioned in the instruction. You should analyze the instruction and determine the most logical target location based on its context. For example, if the instruction is to "clear the inside of the fridge," and the class list includes a kitchen counter, it should also be considered. The kitchen counter is the most logical place to place the items, as it fits the context of the task.
            - You must consider only class requirements, don't try to look for objects based on spatial relations like close to or on top of. For example 'empty the inside of the fridge' means that the relevent classes are fridge and potential location where to place the items like kitchen counter or table.
            - Your role is to identify the most relevant classes based solely on semantics, without considering spatial relationships. For example, in the instruction "Move objects from inside the cabinet," your focus should be on the cabinet and a suitable target surface for placing the objects, such as a table or the floor. Do not attempt to determine which specific objects are inside the cabinet from the given list.
        
        
        Important: the classes you return should be in the provided list of class objects, don't hallucinate any new class names.
        Very important: You for every object class to be moved you must return the most logical target class where it is going to be placed relative to with wither close to or on top of one of its support surfaces. 
        the final output must be in the xml format xml```<objects>[place the relevent classes here]</objects>```, you must select from this list <objects>{list(set([o[0] for o in promptable_classes]))}</objects>
        
        Very Important:
          - Your task is to return the class names of objects strictly from the provided list. For each object selected, you must include at least one target object class from the list to indicate its placement relative to another object (e.g., near the door, on the table, or on the floor).
           - For each relevent object you must include a target object where it will be placed in reference to.  
        
        Tips: 
            - Relevant classes do not necessarily need to be explicitly mentioned in the instruction. You should analyze the instruction and determine the most logical target location based on its context. For example, if the instruction is to "clear the inside of the fridge," and the class list includes a kitchen counter, it should also be considered. The kitchen counter is the most logical place to place the items, as it fits the context of the task.
            - The target location can be near, close to, on top another object it might be mentioned in the instruction so first analyze it and come up with the most logical focal object. 
        Examples of objects relevent to instruction:    
            Instruction: "Clear the books from the desk."

            Relevant Classes: books, desk, bookshelf (as a logical target location)
            Reason: Books are being moved, the desk is where they are originally placed, and a bookshelf is a reasonable target location.
            Instruction: "Put the groceries away."

            Relevant Classes: groceries, fridge, cabinet
            Reason: Groceries need to be stored, with the fridge and cabinets being logical target locations.
            Instruction: "Remove the dishes from the sink."

            Relevant Classes: dishes, sink, dish rack, kitchen counter
            Reason: Dishes are being moved, and logical places include the dish rack or kitchen counter.
            Instruction: "Organize the toys in the living room."

            Relevant Classes: toys, toy box, shelf
            Reason: Toys are the objects being moved, and toy boxes or shelves are potential target locations.
            Instruction: "Take out the trash."

            Relevant Classes: trash, trash bin, door
            Reason: The trash is being moved, and the bin or outside (door as an exit point) are relevant.
        
        Very important: return classes from the provided list, don't hallucinte classes outside of it. If the instruction indicates prular objects you ust retrieve the closest from the set  <objects>{list(set([o[0] for o in promptable_classes]))}</objects>.
        
        
          """
    if type == 'relevent ids':
        objects_details = ""
        for obj in relevent_classes:
            colors = obj.colors[:3]
            materials = obj.material[:3]
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
            obj_description = obj.description[0] if obj.description else obj.obj_name
            obj_details = obj.get_obj_details()
            if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                pred_name = obj.obj_name
            if use_gt and ('cc' in obj.obj_name or obj.obj_name == 'unknown'):
                continue
            objects_details += f"object id: {obj.obj_id}, class: {pred_name}, object colors: {colors}, object material: {materials}, object description: {obj_description} \n"

        prompt = f"""
        You are a helpful assistant responsible for filtering out irrelevant objects based on some color, material, or size that might be mentioned in a given instruction. You will receive a list of objects, each with an ID, color, and material. Your task is to identify and return the object IDs that align with the instruction. If the instruction does not specify any attributes for the desired objects, return all object IDs.

        Additionally, some objects serve as target locations for placing other objects according to the instruction. If no specific attributes are mentioned for these target objects, they should be retained.
        If no color or material are requested in the instructions return all IDs.


        Note: please select object that need to be moved as well as target objects.
        Objects list with their object ids, class name, list of major colors, list of materials:

        {objects_details}

        Instruction is :
        {instruction}

        Please end your thinking with <relevant_ids>[place the list of ids which align with the instruction here]</relevant_ids>

        Tips: 
            - Relevant classes do not necessarily need to be explicitly mentioned in the instruction. You should analyze the instruction and determine the most logical target location based on its context. For example, if the instruction is to "clear the inside of the fridge," and the class list includes a kitchen counter, it should also be considered. The kitchen counter is the most logical place to place the items, as it fits the context of the task.
            - You must consider only semantic attributes like color, material, and specific object characteristics as filtering requirements, don't try to look for objects based on spatial relations like close to or on top of. For example 'empty the inside of the fridge' means that the relevent classes are fridge and potential location where to place the items like kitchen counter or table.

        Very Important:
            - Don't filter any object out unless a color or material attribute is requested.
            - Your task is to return the IDs of objects strictly from the provided list. For each object selected, you must include at least one target object ID from the list to indicate its placement relative to another object (e.g., near the door, on the table, or on the floor).

        Examples of objects relevent to instruction:
            Instruction: "Clear the books from the desk."

            Relevant Classes: books, desk, bookshelf (as a logical target location)
            Reason: Books are being moved, the desk is where they are originally placed, and a bookshelf is a reasonable target location.
            Instruction: "Put the groceries away."

            Relevant Classes: groceries, fridge, cabinet
            Reason: Groceries need to be stored, with the fridge and cabinets being logical target locations.
            Instruction: "Remove the dishes from the sink."

            Relevant Classes: dishes, sink, dish rack, kitchen counter
            Reason: Dishes are being moved, and logical places include the dish rack or kitchen counter.
            Instruction: "Organize the toys in the living room."

            Relevant Classes: toys, toy box, shelf
            Reason: Toys are the objects being moved, and toy boxes or shelves are potential target locations.
            Instruction: "Take out the trash."

            Relevant Classes: trash, trash bin, door
            Reason: The trash is being moved, and the bin or outside (door as an exit point) are relevant.
            
        
        You must keep only {pc_handler.number_of_samples} target location per object to be move, for example to move a vase from its location it can be moved toon top of a table, floor, kitchen counter etc, you must keep only {pc_handler.number_of_samples}
        """ 
    if type == 'relevent relations':
        objects_details = ""
        for obj in relevent_ids:
            colors = obj.colors[:3]
            materials = obj.material[:3]
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
            obj_description = obj.description[0] if obj.description else obj.obj_name
            obj_details = obj.get_obj_details(get_position_and_orientation=True)
            if (obj.obj_name != 'unknown') and ('cc' not in obj.obj_name):
                pred_name = obj.obj_name
            if use_gt and ('cc' in obj.obj_name or obj.obj_name == 'unknown'):
                continue
            objects_details += f"object id: {obj.obj_id}, Dimensions: (dx = {obj_details['dimensions'][0]}, dy = {obj_details['dimensions'][1]}, dy = {obj_details['dimensions'][2]}), class: {pred_name}, object colors: {colors}, object material: {materials}, object description: {obj_description} \n"

            
        prompt = f"""
            You are a Helpful Assistant. Your task is to determine how an object should be retrieved based on the given instructions. Do not try to make a plan for placing the objects. You must focus only on understanding whether retrieval involves relations or not.
                Guideline for Identifying the Object to Move and Retrieval Type

                    Identify the Object to Move
                        Look for action verbs like move, place, put to determine what is being acted upon.
                        Example: "Move the chair." (Chair is the object to move).

                    Check for Relations
                        If a prepositional phrase (e.g., next to the table) describes the destination, it is retrieval without relations ("Move the chair next to the table.").
                        If the phrase describes the current position, it is retrieval with relations ("Move the chair that is next to the table.").

                    Determine Retrieval Type
                        Without Relations: The object is retrieved by intrinsic properties (e.g., color, type).
                        With Relations: The object is retrieved using another object as a reference.

                    Edge Cases
                        If no reference is mentioned, assume retrieval without relations.
                        Multiple objects should be checked for independent or dependent relationships.

                Your role is to analyze retrieval type, not to decide where to place objects.

            You will receive a list of objects, each with details such as their IDs, colors, and materials. The instruction provided may ask for objects relative to others in this list. Your task is to analyze the instruction and identify the objects that can serve as references to locate the target objects.

            For example:
                If the instruction is: "Could you please empty the kitchen counter, then move items from the fridge to the kitchen counter?", the fridge has a relation "on top" with all possible surfaces, and the kitchen counter is also referenced as "on top." In this case, return the relevant object IDs based on these relationships.

            Objects list with their object ids, class name, list of major colors, list of materials:

            {objects_details}

            Instruction is :
            {instruction}

            Please end your thinking with the following xm format 
            ```xml
            <objects>
                <object>
                    <id>[please the object id here]</id>
                    <surface_ids>[please place the desired surface ids here seperated by a comma]</surface_ids>
                    <relation>[please place the required relations here seperated by comma, you must choose from 'on top', 'facing', 'against wall']</relation>
                </object>            
            </objects>

                        ```

            Important regarding surface IDs: A surface is the area where objects can be placed in an object, each object has surfaces with several IDs from 0 and up, the surface with the lowest elevation has the highest ID.
            Example for surfaces: if there are three surfaces with ids 0,1,2,3 the lowest id corresponds to the top surface in this case ID 0 where the highest ID corresponded to the lowest surface in this case 3.

            Examples that can help you understanding if the instruction requires retrieving with spatial constraints or not: 
                Retrieval with Relations (Spatial)
                    Instruction: "Clear the top of the cabinet."

                    Relevant Object: cabinet (ID X), Surface ID: 0, Relation: on top
                    Reason: The instruction specifies "top of the cabinet," meaning objects must be retrieved from surface 0.
                    Instruction: "Empty the inside of the fridge."

                    Relevant Object: fridge (ID Y), Surface IDs: 1, 2, 3, ..., Relation: on top
                    Reason: "Inside" implies retrieving objects from internal shelves (surfaces 1 and above).
                    Instruction: "Remove the books from the bookshelf."

                    Relevant Object: bookshelf (ID Z), Surface IDs: 1, 2, 3, ..., Relation: on top
                    Reason: Books are being retrieved from multiple internal shelves, not just the top.
                
                Retrieval Without Relations (Non-Spatial)
                    Instruction: "Move the chairs."

                    Relevant Object: chairs (all instances)
                    Reason: No spatial reference, just moving all chairs.
                    Instruction: "Rearrange the whiteboards."

                    Relevant Object: whiteboard (all instances)
                    Reason: The instruction does not specify retrieval based on another object.
                    Instruction: "Gather all laptops."

                    Relevant Object: laptops (all instances)
                    Reason: The instruction is about retrieving all laptops without spatial constraints.
            Hint how to address this task assigned to you: First, evaluate the instruction to determine whether any objects need to be retrieved in relation to others. If no objects are to be retrieved in relation to others, return an empty XML tag as follows: xml<objects></objects>.


        """ 
    return prompt


