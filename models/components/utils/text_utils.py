import numpy as np
import ast
import re
import time
from collections import Counter
import ast
import operator
import math

def most_frequent(lst):
    count = Counter(lst)
    return count.most_common(1)[0][0]  # Returns the most frequent value
    
def extract_text_between_tags(text, tag):
    pattern = f'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


# Define supported operators

OPS = {
    ast.Add: lambda x, y: x + y,
    ast.Sub: lambda x, y: x - y,
    ast.Mult: lambda x, y: x * y,
    ast.Div: lambda x, y: x / y,
    ast.Mod: lambda x, y: x % y,
    ast.Pow: lambda x, y: x ** y,
    ast.BitXor: lambda x, y: x ^ y,
}



def evaluate_expr(expr):
    """Evaluate a single mathematical expression safely."""
    def _eval(node):
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Handle unary operations (like -1)
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise ValueError("Unsupported unary operation")
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            arg = _eval(node.args[0])
            if func_name == 'cos':
                return math.cos(math.radians(arg))
            elif func_name == 'sin':
                return math.sin(math.radians(arg))
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Num):
            return node.n
        else:
            raise ValueError("Unsupported expression")

    tree = ast.parse(expr, mode='eval').body
    return _eval(tree)

def parse_and_compute(input_list):
    """Parse and compute a list of mathematical expressions."""
    # Remove brackets and split elements
    input_list = str(input_list)
    input_list = input_list.replace("x", '').replace("y", '').replace("z", '').replace("=", '').replace("'x'", '').replace("'y'", '').replace("'z'", '').replace(':', '').replace('degrees', '').replace('degree', '').replace('{', '').replace('}', '')
    input_list = input_list
    input_list = input_list.strip('[]')
    input_list = input_list.strip('()')
    elements = [elem.strip() for elem in input_list.split(',')]

    # Evaluate each element
    results = []
    for elem in elements:
        try:
            # Compute value if it contains an operation
            # results.append(evaluate_expr(elem) if any(op in elem for op in '+-*/%') else float(elem))
            try:
                results.append(float(elem.replace('degrees', '').replace('"', '').replace("'", '').replace('degree', '').replace('°', '').replace('(', '').replace(')', '')))
            except:
                results.append(evaluate_expr(elem.replace('degrees', '').replace('"', '').replace("'", '').replace('degree', '').replace('°', '')) if any(op in elem for op in '+-*/%') or 'cos' in elem or 'sin' in elem else float(elem))

        except Exception as e:
            raise ValueError(f"Error parsing element '{elem}': {e}")

    return results


