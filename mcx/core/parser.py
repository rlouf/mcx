import ast
import inspect
from types import FunctionType
from typing import Any, Dict, List, Union

import astor

from mcx.core.graph import GraphicalModel


def parse_definition(model: FunctionType, namespace: Dict):
    """Build a graph representation of the model from its definition.

    Arguments
    ---------
    model
        The function that contains the model's definition.
    namespace
        The global namespace where the model is going to be invoked.

    Returns
    -------
    A GraphicalModel object.
    """
    src = inspect.getsource(model)
    tree = ast.parse(src)
    return Parser().visit(tree)


class Parser(ast.NodeVisitor):
    """Recursively parse the model definition's AST and translate it to a
    graphical model.
    """

    def __init__(self):
        self.model = GraphicalModel()

    def generic_visit(self, node):
        node_type = type(node).__name__
        raise SyntaxError("Method to process {} not specified".format(node_type))

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

        self.visit_Model(model_fn)
        return self.model

    #
    # MCX-specfic visitors
    #

    def visit_Model(self, node: ast.FunctionDef) -> None:
        """Visit the function in which the model is defined.

        Raises
        ------
        SyntaxError
            If the function's body contains unsupported constructs, i.e.
            anything else than an assignment, an expression or return.
        """
        self.visit_ModelDef(node)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self.visit_Deterministic(stmt)
            elif isinstance(stmt, ast.Expr):
                self.visit_Expr(stmt)
            elif isinstance(stmt, ast.Return):
                self.visit_Return(stmt)
            else:
                raise SyntaxError(
                    "Only variable, random variable assignments and transformations are currently supported"
                )

    def visit_ModelDef(self, node: ast.FunctionDef) -> None:
        """Record the model's name and its arguments.
        """
        self.model.graph["name"] = node.name
        for arg in node.args.args:
            self.model.add_argument(arg.arg)

    def visit_Deterministic(self, node: ast.Assign) -> None:
        """Visit and add deterministic variables to the graphical model.

        Deterministic expression can be of two kinds:

        - Constant assignments;
        - Transformation of existing variables.

        Since constants can also be the result of a call of the form
        `np.array([0, 1])`, we need to walk down the assignments' values. If
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
                value = node.value
                self.model.add_variable(var_name, value=value)
            elif isinstance(node.value, ast.Num):
                value = node.value
                self.model.add_variable(var_name, value=value)
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
                "Require a name on the left-hand-side of a deterministic variable assignment, got {}".format(
                    target
                )
            )

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.BinOp):
            self.visit_BinOp(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.visit_RandAssign(node)

    def visit_RandAssign(self, node: ast.BinOp) -> None:
        """Visit a random variable assignment, and add a new random node to the
        graph.

        Random variable assignments are distinguished from deterministic assignments
        by the use of the `@` operator.

        Raises
        ------
        SyntaxError
            If there is no variable name on the left-hand-side of the `@` operator.
        SyntaxError
            If the right-hand-side is not a function call.
            #TODO: check that the class being initialized is a Distribution instance.
        """
        if not isinstance(node.left, ast.Name):
            raise SyntaxError(
                "You need a variable name on the left of a random variable assignment, got {}".format(
                    node.left
                )
            )
        if not isinstance(node.right, ast.Call):
            raise SyntaxError(
                "Statements on the right of the `@` operator must be distribution initialization, got {}".format(
                    node.right
                )
            )
        name = node.left.id
        distribution = node.right
        args = self.visit_Call(node.right)
        self.model.add_randvar(name, distribution, args)

    def visit_Call(self, node: ast.Call) -> List[Union[str, float, int]]:
        return self.visit_Arguments(node.args)

    def visit_Arguments(self, args: List[Any]) -> List[Union[str, float, int]]:
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
        arguments = []
        for arg in args:
            if isinstance(arg, ast.Name):
                arguments.append(arg.id)
            elif isinstance(arg, ast.Constant):
                arguments.append(arg.value)
            elif isinstance(arg, ast.Num):
                arguments.append(arg.n)
            else:
                raise SyntaxError(
                    "Expected a random variable of a constant to initialize distribution, got {} instead.\n"
                    "Maybe you are trying to initialize a distribution directly, or call a function inside the "
                    "distribution initialization. While this would be a perfectly legitimate move, it is currently "
                    "not supported in mcx. Use an intermediate variable instead: \n\n"
                    "Do not do `x @ Normal(Normal(0, 1), 1)` or `x @ Normal(my_function(10), 1)`, instead do "
                    " `y @ Normal(0, 1) & x @ Normal(y, 1)` and `y = my_function(10) & x @ Normal(y, 1)`".format(
                        astor.code_gen.to_source(arg)
                    )
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
                "Expected the generative model to return a (random) variable or a tuple of (random) variables, got {}".format(
                    node.value
                )
            )


def find_variable_arguments(node) -> List[str]:
    """ Walk down the Abstract Syntax Tree of the right-hand-side of the
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
