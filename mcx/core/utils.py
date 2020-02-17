import ast
from typing import Dict, List, Optional


def read_object_name(node: ast.AST, name: Optional[List[str]] = None) -> str:
    """Parse the object's (class or function) name from the right-hand size of an
    assignement nameession.

    The parsing is done recursively to recover the full import path of the
    imported names. This step is only an operation on strings: we do not check
    whether the names are indeed present in the namespace.

    Args:
        node: The right-hand-side of the assignment node.
        name: The parts of the name that have been parsed.

    Returns:
        The full path to the object on the right-hand-side of the assignment.

    Raises:
        ValueError: If the next node in the recursion is neither a variable
            (ast.Name) nor and attribute.

    Examples:
        When the name is imported directly:

        >>> read_object_name(ast.Name(id='Exponential'))
        Exponential

        When it is imported from a submodule. Here for `mcmx.distributions.Normal`

        >>> read_object_name(
        ...     ast.Attribute(value=ast.Attribute(value=ast.Name(id='mcmx'), attr='distributions'), attr='Normal')
        ... )
        mcmx.distributions.Normal
    """
    if not name:
        name = []

    if isinstance(node, ast.Name):
        name.insert(0, node.id)
        return ".".join(name)
    elif isinstance(node, ast.Attribute):
        name.insert(0, node.attr)
        new_node = node.value
        return read_object_name(new_node, name)
    else:
        raise ValueError(
            "Error while getting the right-hand-side object's name: expected `ast.Name` or `ast.Attribute`, got {}".format(
                node
            )
        )


def relabel_arguments(value_node: ast.AST, mapping: Dict):
    """ Walk down the Abstract Syntax Tree of the right-hand-side of the
    assignment to relabel the arguments.

    Returns
    -------
    A list of variable names, default to an empty list.
    """
    for node in ast.walk(value_node):
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name):
                var_name = node.left.id
                if var_name in mapping:
                    node.left.id = mapping[var_name]
            if isinstance(node.right, ast.Name):
                var_name = node.right.id
                if var_name in mapping:
                    node.right.id = mapping[var_name]
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    var_name = arg.id
                    if var_name in mapping:
                        arg.id = mapping[var_name]
