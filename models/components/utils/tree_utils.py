"""
Tree and Graph Utilities

Functions for managing hierarchical relationships and dependency ordering
in scene graphs. Used when transformations must be applied in correct order
(parents before children).
"""

import numpy as np
from collections import defaultdict, deque


def sort_tree_relations_with_indices(tree):
    """
    Sort parent-child relations using topological ordering.
    
    Ensures parents are processed before children, critical for applying
    transformations hierarchically (e.g., moving a table also moves items on it).
    
    Algorithm: Kahn's topological sort (BFS-based)
        1. Build graph of dependencies (parent -> children)
        2. Count incoming edges (in-degree) for each node
        3. Process nodes with no dependencies first
        4. Remove processed nodes and update in-degrees
        5. Continue until all nodes processed
    
    Args:
        tree (list of tuples): List of (child, parent) relations.
                              E.g., [(cup, table), (book, shelf)]
    
    Returns:
        tuple: (sorted_relations, sorted_indices)
            - sorted_relations: Relations ordered parent-first
            - sorted_indices: Original indices of sorted relations
            
    Example:
        >>> tree = [(3, 1), (2, 1), (4, 2)]  # 1->2->4, 1->3
        >>> sorted_rels, indices = sort_tree_relations_with_indices(tree)
        >>> # Returns: [(3, 1), (2, 1), (4, 2)] (parent 1 before children)
    """
    # Map each relation to its original index for tracking
    index_map = {pair: i for i, pair in enumerate(tree)}

    # Build adjacency list (parent -> [children]) and in-degree count
    graph = defaultdict(list)
    in_degree = defaultdict(int)

    for child, parent in tree:
        graph[parent].append(child)
        in_degree[child] += 1
        if parent not in in_degree:
            in_degree[parent] = 0  # Root nodes have in-degree 0

    # Start with nodes that have no parents (in-degree 0)
    queue = deque(sorted(node for node in in_degree if in_degree[node] == 0))
    sorted_relations = []
    sorted_indices = []

    # Process nodes level by level
    while queue:
        parent = queue.popleft()
        
        # Process all children of current parent
        for child in sorted(graph[parent]):
            pair = (child, parent)
            sorted_relations.append(pair)
            sorted_indices.append(index_map[pair])
            
            # Decrease in-degree and add to queue if ready
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return sorted_relations, sorted_indices