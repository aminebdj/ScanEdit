"""
Color Generation Utilities

Functions for generating color palettes for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_vibrant_cool_colors(n):
    """
    Generate n vibrant cool colors (blue, cyan, purple, green shades).
    
    Uses HSV color space to ensure high saturation and brightness while
    constraining hue to cool color range (180-300 degrees).
    
    Args:
        n (int): Number of colors to generate
    
    Returns:
        list of tuple: RGB colors normalized to [0, 1]
        
    Example:
        >>> colors = generate_vibrant_cool_colors(5)
        >>> for color in colors:
        ...     # Use color for Open3D: mesh.paint_uniform_color(color)
        
    Note:
        Colors vary within cool spectrum to provide visual distinction
        between different objects in scene visualization.
    """
    colors = []
    for i in range(n):
        # Hue range: 180-300 degrees (cool colors)
        hue = np.linspace(180/360, 300/360, n)[i]
        
        # High saturation (0.8-1.0) for vibrancy
        saturation = 0.8 + 0.2 * np.random.rand()
        
        # High brightness (0.8-1.0)
        value = 0.8 + 0.2 * np.random.rand()
        
        # Convert HSV to RGB
        rgb = plt.cm.hsv(hue)[:3]
        rgb = tuple(np.array(rgb) * value)
        
        colors.append(rgb)

    return colors


def generate_color_dict(n):
    """
    Generate dictionary of simple named colors with RGB values.
    
    Returns predefined common colors (red, blue, green, etc.) as a dictionary
    mapping color names to normalized RGB tuples. Useful for consistent
    object labeling and visualization.
    
    Args:
        n (int): Number of colors needed (max 14)
    
    Returns:
        dict: {color_name: (r, g, b)} where RGB values are in [0, 1]
        
    Raises:
        ValueError: If n > 14 (exceeds available predefined colors)
        
    Example:
        >>> colors = generate_color_dict(5)
        >>> print(colors)
        >>> # {'red': (1.0, 0.0, 0.0), 'green': (0.0, 0.502, 0.0), ...}
        >>> 
        >>> # Use for visualization
        >>> mesh.paint_uniform_color(colors['red'])
        
    Use case:
        Assigning consistent colors to objects for visualization and debugging
    """
    # Predefined simple color names (most common first)
    SIMPLE_COLORS = [
        "red", "green", "blue", "yellow", "orange", "pink", 
        "purple", "cyan", "black", "gray", "brown", "lime", 
        "teal", "magenta"
    ]
    
    if n > len(SIMPLE_COLORS):
        raise ValueError(
            f"Cannot generate {n} distinct colors. "
            f"Maximum is {len(SIMPLE_COLORS)}."
        )
    
    # Select first n colors
    selected_colors = SIMPLE_COLORS[:n]
    
    # Create dictionary with RGB tuples (rounded to 3 decimals)
    color_dict = {
        color: tuple(round(val, 3) for val in to_rgb(color))
        for color in selected_colors
    }
    
    return color_dict

def top_k_frequent_elements(lst, k):
    """
    Get k most frequent elements in list.
    
    Uses Counter to efficiently find most common elements.
    Useful for finding dominant object types or classes.
    
    Args:
        lst (list): Input list
        k (int): Number of most frequent elements to return
    
    Returns:
        list: K most frequent elements
        
    Example:
        >>> items = ['chair', 'table', 'chair', 'chair', 'desk']
        >>> top = top_k_frequent_elements(items, 2)
        >>> # Returns: ['chair', 'table']
    """
    from collections import Counter
    count = Counter(lst)
    return [item for item, freq in count.most_common(k)]