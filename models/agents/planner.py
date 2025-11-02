"""
Scene Editing Planner

Determines what objects to move and how based on natural language instructions.
Uses LLM reasoning to parse instructions into structured editing plans.

Pipeline:
1. Identify relevant object classes from instruction
2. Select specific object instances by ID
3. Extract spatial relationships
4. Generate placement plan
5. Create hierarchical editing structure
"""

import xmltodict
from models.components.utils.text_utils import extract_text_between_tags
from models.agents.prompts.planner import get_prompt
from models.components.utils.io_utils import load_or_execute


def Planner(pc_handler, text_instruction, root_path, relevent_objects, relevent_relations, load_prev=True, llm=None):
    """
    Generate hierarchical editing plan from natural language instruction.
    
    Analyzes text instruction to determine:
    - Which objects are affected
    - What spatial relationships to enforce
    - How to structure the editing operations
    
    Algorithm:
        1. Extract relevant object classes (e.g., "chair", "table")
        2. Identify specific object IDs from those classes
        3. Parse spatial relationships (e.g., "chair near table")
        4. Generate placement plan with constraints
        5. Structure into hierarchical editing dictionary
    
    Args:
        pc_handler: Scene handler with object information
        text_instruction (str): Natural language editing command
                               (e.g., "Move the chair closer to the table")
        root_path (str): Directory for caching/loading intermediate results
        load_prev (bool): Load cached results if available. Default: True
        llm: Language model instance. If None, uses global reasoningllm
    
    Returns:
        tuple: (hierarchy_dictionary, editing_type)
            - hierarchy_dictionary (dict): Structured editing plan with object
                                          IDs, constraints, and relationships
            - editing_type (str): Type of editing operation 
                                 ('rearrange' or 'remove')
    
    Example:
        >>> handler = SCENE_HANDLER(...)
        >>> instruction = "Move the chair next to the table"
        >>> plan, edit_type = Planner(handler, instruction, "./results")
        >>> print(plan)
        >>> # {'objects': {'object': [{'id': '3', 'constraint': 'near', ...}]}}
    """
    # Use provided LLM or fall back to global
    if llm is None:
        exit()

    # ========================================================================
    # STEP 5: Determine Editing Type
    # ========================================================================
    # Classify as rearrangement or removal operation
    
    editing_type = 'rearrange' if 'remove' not in text_instruction else 'remove'
    
    # ========================================================================
    # STEP 6: Generate Placement Plan
    # ========================================================================
    # Create detailed plan with constraints and target positions
    
    instruction_type = 'placement plan'
    prompt = get_prompt(
        pc_handler=pc_handler,
        instruction=text_instruction,
        relevent_ids=relevent_objects,
        type=instruction_type,
        relevent_relations_ids=relevent_relations,
        editing_type=editing_type
    )
    
    placement_plan_answer = load_or_execute(
        root_path, 
        'placement_plan.yaml',
        load_prev, 
        llm, 
        prompt
    )
    
    # Extract placement plan from response
    placement_plan = extract_text_between_tags(placement_plan_answer, 'placement_plan')[0]

    # ========================================================================
    # STEP 7: Generate Hierarchical Structure
    # ========================================================================
    # Convert placement plan into hierarchical editing dictionary
    
    instruction_type = 'generate group'
    prompt = get_prompt(
        pc_handler=pc_handler,
        placement_plan=placement_plan,
        dependency_plan='None',
        relevent_ids=relevent_objects,
        type=instruction_type,
        relevent_relations_ids=relevent_relations
    )
    
    hierarchy_answer = load_or_execute(
        root_path, 
        'hierarchy.yaml',
        load_prev, 
        llm, 
        prompt
    )
    
    # ========================================================================
    # STEP 8: Clean and Parse XML Response
    # ========================================================================
    # LLM may wrap XML in various tags - extract the actual XML content
    
    # Handle different possible tag wrappers
    if '<placement_plan>' in hierarchy_answer:
        hierarchy_answer = '```' + extract_text_between_tags(
            hierarchy_answer, 'placement_plan'
        )[0] + '```'
    
    if '<root>' in hierarchy_answer:
        hierarchy_answer = '```' + extract_text_between_tags(
            hierarchy_answer, 'root'
        )[0] + '```'
    
    if '<objects>' in hierarchy_answer:
        hierarchy_answer = '```<objects>' + extract_text_between_tags(
            hierarchy_answer, 'objects'
        )[0] + '</objects>```'
    
    if '<xml>' in hierarchy_answer:
        hierarchy_answer = '```<objects>' + extract_text_between_tags(
            hierarchy_answer, 'xml'
        )[0] + '</objects>```'
    
    # Parse XML to dictionary
    hierarchy_dictionary = xmltodict.parse(
        hierarchy_answer.split('```')[1].replace('xml', '').replace('\n', '')
    )
    
    # Normalize structure - ensure 'objects' key exists
    if 'object' not in list(hierarchy_dictionary.keys())[0]:
        hierarchy_dictionary_ = {}
        hierarchy_dictionary_['objects'] = list(hierarchy_dictionary.values())[0]
        hierarchy_dictionary = hierarchy_dictionary_
    
    # Extract objects from structure
    if 'objects' in hierarchy_dictionary.keys():
        hierarchy_dictionary = hierarchy_dictionary['objects']
    
    # Always return 'rearrange' type (removal handled separately)
    return hierarchy_dictionary, 'rearrange'


def extract_removal_ids(placement_plan_answer):
    """
    Extract object IDs marked for removal from placement plan.
    
    Helper function for processing removal operations.
    
    Args:
        placement_plan_answer (str): LLM response containing removal IDs
    
    Returns:
        list of int: Object IDs to remove
        
    Example:
        >>> answer = "<remove>[3, 5, 7]</remove>"
        >>> ids = extract_removal_ids(answer)
        >>> print(ids)  # [3, 5, 7]
    """
    remove_text = extract_text_between_tags(placement_plan_answer, 'remove')[0]
    remove_text = remove_text.strip('[').strip(']').strip('(').strip(')')
    remove_ids = [int(idx) for idx in remove_text.split(',')]
    return remove_ids