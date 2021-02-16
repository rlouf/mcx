import ast
import inspect
import re
import textwrap
from typing import Any, Callable, Dict, List, Optional, Union

import mcx
from mcx.compiler.graph import GraphicalModel


def parse_definition(model: Callable, namespace: Dict) -> GraphicalModel:
    """Build a graph representation from the model's definition.

    To simplify the parsing of the Abstract Syntax Tree of the model definition
    we substitute the combination `<~` of operators for the `is` operator. This
    choice was inspired by the yaps [1]_ library.

    Parameters
    ----------
    model
        The function that contains the model's definition.
    namespace
        The global namespace where the model is going to be invoked.

    Returns
    -------
    The internal representation of the model as a graph.

    References
    ----------
    .. [1]: yaps, a surface language for Stan with python syntax. Source
            code available at https://github.com/IBM/yaps. Distributed under
            the Apache 2.0 Licence.

    """
    source = inspect.getsource(model)
    source = re.sub(r"<~", "is", source, re.X)
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    return ModelParser(namespace).visit(tree)


class ModelParser(ast.NodeVisitor):
    """Recursively parse the model definition's AST and translate it to a
    graphical model.

    """

    def __init__(self, namespace):
        self.model = GraphicalModel()
        self.namespace = namespace

    def generic_visit(self, node):
        node_type = type(node).__name__
        raise SyntaxError(f"Method to process {node_type} not specified")

    def visit_Module(self, node: ast.Module) -> GraphicalModel:
        """Parsing the source code into an Abstract Syntax Tree returns an
        `ast.Module` object. We check that the object being parsed is indeed a
        single function and pass on to another method to parse the function.

        Returns
        -------
        A graphical model instance that contains the graphical representation
        of the model.

        Raises
        ------
        SyntaxError
            If the module's body is empty or contains more than one object.
        SyntaxError
            If the module's body does not contain a funtion definition.

        """
        if len(node.body) != 1:
            raise SyntaxError("You must pass a single model definition.")

        model_fn = node.body[0]
        if not isinstance(model_fn, ast.FunctionDef):
            raise SyntaxError("The model must be defined inside a function")

        self.visit_model(model_fn)
        return self.model

    #
    # MCX-specfic visitors
    #

    def visit_model(self, node: ast.FunctionDef) -> None:
        """Visit the function in which the model is defined.

        Raises
        ------
        SyntaxError
            If the function's body contains unsupported constructs, i.e.
            anything else than an assignment, an expression or return.

        """
        self.visit_model_arguments(node)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self.visit_deterministic(stmt)
            elif isinstance(stmt, ast.Expr):
                self.visit_Expr(stmt)
            elif isinstance(stmt, ast.Return):
                self.visit_Return(stmt)
            else:
                raise SyntaxError(
                    "Only variable, random variable assignments and transformations are currently supported"
                )

    def visit_model_arguments(self, node: ast.FunctionDef) -> None:
        """Record the model's name and its arguments."""
        self.model.graph["name"] = node.name

        argument_names: List[str] = []
        for arg in node.args.args:
            argument_names.append(arg.arg)

        num_arguments = len(argument_names)
        default_values: List[Union[ast.expr, None]] = []
        for default in node.args.defaults:
            default_values.append(default)
        default_values = (num_arguments - len(default_values)) * [None] + default_values  # type: ignore

        for name, value in zip(argument_names, default_values):
            self.model.add_argument(name, value)

    def visit_deterministic(self, node: ast.Assign) -> None:
        """Visit and add deterministic variables to the graphical model.

        Deterministic expression can be of two kinds:

        - Constant assignments;
        - Transformation of existing variables.

        Since constants can also be the result of a call of the form
        `jnp.array([0, 1])`, we need to walk down the assignments' values. If
        any `ast.Name` node is find, the assignment is a transformation
        otherwise it is a constant.

        Raises
        ------
        SyntaxError
            If several variables are being assigned in one statement.
        SyntaxError
            If this is not a variable assignment.
            #TODO: check that the functions being called are in scope.

        """
        if len(node.targets) != 1:
            raise SyntaxError("You can only assign one variable per statement.")

        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            if isinstance(node.value, ast.Constant):
                constant_value = node.value
                self.model.add_variable(var_name, value=constant_value)
            elif isinstance(node.value, ast.Num):
                num_value = node.value
                self.model.add_variable(var_name, value=num_value)
            else:
                arg_names = find_variable_arguments(node)
                if arg_names:
                    expression = node.value
                    self.model.add_transformation(var_name, expression, arg_names)
                else:
                    value = node.value
                    self.model.add_variable(var_name, value)
        else:
            raise SyntaxError(
                "Require a name on the left-hand-side of a deterministic "
                f"variable assignment, got {target}"
            )

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Compare):
            self.visit_Compare(node.value)

    def visit_Compare(self, node: ast.Compare) -> None:
        if isinstance(node.ops[0], ast.Is):
            self.visit_RandAssign(node)

    def visit_RandAssign(self, node: ast.Compare) -> None:
        """Visit a random variable assignment, and add a new random node to the
        graph.

        Random variable assignments are distinguished from deterministic assignments
        by the use of the `<~` operator.

        Raises
        ------
        SyntaxError
            If there is no variable name on the left-hand-side of the `<~` operator.
        SyntaxError
            If the right-hand-side is not a function call.
            #TODO: check that the class being initialized is a Distribution instance.

        """
        if not isinstance(node.left, ast.Name):
            raise SyntaxError(
                "You need a variable name on the left of a random variable assignment"
                f", got {node.left}"
            )
        if not isinstance(node.comparators[0], ast.Call):
            raise SyntaxError(
                "Statements on the right of the `<~` operator must be distribution "
                f"initialization, got {node.comparators[0]}"
            )
        name = node.left.id
        distribution = node.comparators[0]
        args = self.visit_Call(node.comparators[0])

        # To allows model composition, whenever a `mcx` model appears at the
        # right-hand-side of a `<~` operator we merge its graph with the current
        # model's graph
        dist_path = read_object_name(distribution.func)
        dist_obj = eval(dist_path, self.namespace)
        if isinstance(dist_obj, mcx.model):
            print(args)
            self.model = self.model.merge_models(name, dist_obj.graph, args)
        else:
            self.model.add_randvar(name, distribution, args)

    def visit_Call(self, node: ast.Call) -> List[Union[str, int, float, complex]]:
        return self.visit_Arguments(node.args)

    def visit_Arguments(self, args: List[Any]) -> List[Union[str, float, int, complex]]:
        """Visits and returns the arguments used to initialize the distribution.

        Returns
        -------
        A list of the names or values of the arguments passed to the distribution.

        Raises
        ------
        SyntaxError
           If the distribution is initialized with anything different from a constant
           or a previously defined variable.

        """
        arguments: List[Union[str, float, int, complex]] = []
        for arg in args:
            if isinstance(arg, ast.Name):
                arguments.append(arg.id)
            elif isinstance(arg, ast.Constant):
                arguments.append(arg.value)
            elif isinstance(arg, ast.Num):
                arguments.append(arg.n)
            else:
                raise SyntaxError(
                    "Expected a random variable of a constant to initialize "
                    f"distribution, got {astor.code_gen.to_source(arg)} instead.\n"
                    "Maybe you are trying to initialize a distribution directly, "
                    "or call a function inside the distribution initialization. "
                    "While this would be a perfectly legitimate move, it is currently "
                    "not supported in mcx. Use an intermediate variable instead: \n\n"
                    "Do not do `x <~ Normal(Normal(0, 1), 1)` or "
                    "`x <~ Normal(my_function(10), 1)`, instead do "
                    "`y <~ Normal(0, 1) & x <~ Normal(y, 1)` and "
                    "`y = my_function(10) & x <~ Normal(y, 1)`"
                )
        return arguments

    def visit_Return(self, node):
        """Visits the `return` expression of the model definition and mark the
        corresponding variables as returned in the graphical model.

        Raises
        ------
        SyntaxError
            If the model does not return any variable.

        """
        if isinstance(node.value, ast.Tuple):
            for var in node.value.elts:
                if isinstance(var, ast.Name):
                    self.model.mark_as_returned(var.id)
        elif isinstance(node.value, ast.Name):
            self.model.mark_as_returned(node.value.id)
        else:
            raise SyntaxError(
                "Expected the generative model to return a (random) variable or a tuple"
                f"of (random) variables, got {node.value}"
            )


def find_variable_arguments(node) -> List[str]:
    """Walk down the Abstract Syntax Tree of the right-hand-side of the
    assignment to find the named variables, if any.

    Returns
    -------
    A list of variable names, default to an empty list.

    """
    var_names = []
    for node in ast.walk(node.value):
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name):
                var_names.append(node.left.id)
            if isinstance(node.right, ast.Name):
                var_names.append(node.right.id)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    var_names.append(arg.id)
    return var_names


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
            "Error while getting the right-hand-side object's name: expected `ast.Name`"
            f"or `ast.Attribute`, got {node}"
        )
