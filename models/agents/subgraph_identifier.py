import xmltodict
from models.components.utils.text_utils import extract_text_between_tags
from models.agents.prompts.subgraph_identification import get_prompt
from models.components.utils.io_utils import load_or_execute

def Subgraph_Identifier(pc_handler, text_instruction, root_path, load_prev=True, llm=None):
    # ========================================================================
    # STEP 1: Identify Relevant Object Classes
    # ========================================================================
    # Extract which types of objects are mentioned in the instruction
    # Example: "Move the chair near the table" -> ["chair", "table"]
    if llm is None:
        exit()
    instruction_type = 'relevent classes'
    prompt = get_prompt(
        pc_handler=pc_handler,
        instruction=text_instruction,
        type=instruction_type
    )
    
    answer = load_or_execute(
        root_path, 
        'relevent_classes.yaml',
        load_prev, 
        llm, 
        prompt
    )
    
    # Parse LLM response to extract class names
    relevent_classes = extract_text_between_tags(answer, 'objects')[0]
    relevent_classes = relevent_classes.replace('[', '').replace(']', '').replace("'", '')
    relevent_classes = [c.strip().lower() for c in relevent_classes.split(',')]
    
    # ========================================================================
    # STEP 2: Filter Objects by Class
    # ========================================================================
    # Keep only objects matching the relevant classes
    
    pruned_tree = []
    for obj in pc_handler.objects:
        # Skip corrupted or invalid objects
        if obj.obj_name == 'corrupt':
            continue
        
        # Determine object name (use prediction if main name is unknown)
        if 'cc' not in obj.obj_name and obj.obj_name != 'unknown':
            pred_name = obj.obj_name
        else:
            pred_name = obj.pred_obj_name[0] if obj.pred_obj_name else obj.obj_name
        
        # Add to pruned tree if class matches
        if pred_name in relevent_classes:
            pruned_tree.append(obj)
    
    # ========================================================================
    # STEP 3: Identify Specific Object IDs
    # ========================================================================
    # From the filtered objects, select specific instances mentioned
    # Example: "the red chair" -> ID 5, "the large table" -> ID 12
    
    instruction_type = 'relevent ids'
    prompt = get_prompt(
        pc_handler=pc_handler,
        instruction=text_instruction,
        relevent_classes=pruned_tree,
        type=instruction_type
    )
    
    answer = load_or_execute(
        root_path, 
        'relevent_ids.yaml', 
        load_prev, 
        llm, 
        prompt
    )
    
    # Parse object IDs (try both spellings of "relevant")
    try:
        relevent_ids = extract_text_between_tags(answer, 'relevant_ids')[0]
    except:
        relevent_ids = extract_text_between_tags(answer, 'relevent_ids')[0]
    
    # Clean and convert to integers
    relevent_ids = relevent_ids.strip('[]').strip('()').replace("'", '')
    relevent_ids = [
        int(id_txt.replace('[', '').replace(']', '').strip().lower()) 
        for id_txt in relevent_ids.split(',')
    ]
    
    # Get object instances by ID
    relevent_objects = [pc_handler.objects[id] for id in relevent_ids]
    
    # ========================================================================
    # STEP 4: Extract Spatial Relationships
    # ========================================================================
    # Parse relationships between objects (e.g., "chair near table")
    
    instruction_type = 'relevent relations'
    prompt = get_prompt(
        pc_handler=pc_handler,
        instruction=text_instruction,
        relevent_ids=relevent_objects,
        type=instruction_type
    )
    
    answer = load_or_execute(
        root_path, 
        'relevent_relations_ids.yaml',
        load_prev, 
        llm, 
        prompt
    )
    
    # Parse XML response
    relevent_relations_ids = xmltodict.parse(
        answer.split('```')[1].replace('xml', '').replace('\n', '')
    )
    
    # Extract object list from XML structure
    relevent_realtions = []
    if 'objects' in relevent_relations_ids.keys():
        relevent_realtions = relevent_relations_ids['objects']
    
    if isinstance(relevent_realtions, dict) and 'object' in relevent_realtions.keys():
        relevent_realtions = relevent_realtions['object']
    
    # Normalize to list if single object
    if isinstance(relevent_realtions, dict):
        relevent_realtions = [relevent_realtions]
    
    # Normalize keys to lowercase
    if isinstance(relevent_realtions, list):
        relevent_realtions = [
            {k.lower(): v for k, v in rel.items()}
            for rel in relevent_realtions
        ]
    else:
        relevent_realtions = []
    return relevent_objects, relevent_realtions
    