"""
Group Management Utilities

Functions for managing and processing groups of related objects,
handling intersections and dependencies between object groups.
"""

import copy


def remove_intersection(groups, apply_trainable_transf_to):
    """
    Remove intersecting elements from object groups.
    
    When objects belong to multiple groups, this resolves conflicts by:
    1. Merging smaller groups into larger ones
    2. Removing duplicate memberships
    3. Maintaining group hierarchy
    
    Algorithm:
        1. Create promptable groups (groups that can be modified)
        2. Sort by size (smaller groups processed first)
        3. For each group pair, merge smaller into larger if overlap exists
        4. Remove elements from larger group if found in smaller
        5. Return cleaned groups without intersections
    
    Args:
        groups (list): List of groups, where each group is a list of object IDs
                      Can contain None for empty groups
        apply_trainable_transf_to (list): List of transformable object IDs per group
    
    Returns:
        list: Cleaned groups with no intersections, same length as input
              Contains None for removed groups
        
    Example:
        >>> groups = [[1, 2], [2, 3], [4, 5]]
        >>> transforms = [[], [], []]
        >>> cleaned = remove_intersection(groups, transforms)
        >>> # Groups with object 2 will be merged
        
    Use case:
        Scene graph construction where objects can't belong to multiple
        transformation groups simultaneously
    """
    # Create promptable groups: [group_id, *members, *transformable_members]
    promptable_groups = [
        [g_id] + g + apply_trainable_transf_to[g_id] 
        for g_id, g in enumerate(groups) 
        if g is not None
    ]
    
    # Sort by size (smaller groups first for merging)
    promptable_groups = sorted(promptable_groups, key=len)
    promptable_groups_copy = copy.deepcopy(promptable_groups)
    
    remove_id = []
    
    # Process each group against larger groups
    for g_id, g in enumerate(promptable_groups):
        for g_id_, g_ in enumerate(promptable_groups[g_id + 1:]):
            # Check if current group ID appears in larger group
            if g[0] in promptable_groups_copy[g_id + g_id_ + 1]:
                # Mark for removal and merge into larger group
                remove_id.append(g_id)
                promptable_groups_copy[g_id + g_id_ + 1] += g
                break
            else:
                # Remove elements from larger group if found in current
                for ele in promptable_groups_copy[g_id + g_id_ + 1]:
                    if ele in g:
                        promptable_groups_copy[g_id + g_id_ + 1].remove(ele)
    
    # Remove merged groups
    for rm_id in set(remove_id):
        promptable_groups_copy.remove(promptable_groups_copy[rm_id])
    
    # Validate no intersections remain (debug check)
    for g in promptable_groups_copy:
        for g_i in g:
            if any([g_i in g_ for g_ in promptable_groups_copy if g_ != g]):
                print(f'Warning: intersection found: {g}')
    
    # Reconstruct groups in original format
    new_groups = [None for _ in range(len(groups))]
    for g_new in promptable_groups_copy:
        group_id = g_new[0]
        members = list(set(g_new[1:]))  # Remove duplicates
        new_groups[group_id] = members
    
    return new_groups