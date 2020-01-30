import ast
from typing import Callable, List, Optional, Union

import astor
from jax import numpy as np
import networkx as nx

from mcx.distributions import Distribution


# TODO
# [x] Model arguments
# [ ] Deterministic assignments
# [ ] RandVar

class Argument(object):
    name: str
    default: Optional[Union[int, float]] = None


class RandVar(object):
    name: str
    distribution: Distribution
    args: List[Union[int, float, np.DeviceArray]]
    returned: bool


class Var(object):
    name: str
    expression: Optional[Callable]
    value: None
    returned: bool



class Parser(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
        self.distributions = []
        self.vars = []
        self.randvars = []

        self.model = nx.DiGraph()

    def generic_visit(self, node):
        node_type = type(node).__name__
        raise SyntaxError("Method to process {} not specified".format(node_type))

    def visit_Module(self, node):
        assert len(node.body) == 1
        model = node.body[0]
        if not isinstance(model, ast.FunctionDef):
            raise SyntaxError("The model must be defined inside a function")
        self.visit_Model(model)
        if not self.returned:
            raise SyntaxError(
                "Expected a returned value in the definition of the generative model, got None."
            )
        print(self.vars, self.randvars)
        return (
            self.model,
            self.functions,
            self.distributions,
            self.vars,
            self.randvars,
        )

    #
    # MCX-specfic visitors
    #

    def visit_Model(self, node: ast.FunctionDef):
        self.visit_ModelDef(node)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                self.visit_Deterministic(stmt)
            elif isinstance(stmt, ast.Expr):
                self.visit_Expr(stmt)
            elif isinstance(stmt, ast.Return):
                self.visit_Return(stmt)

    def visit_ModelDef(self, node: ast.FunctionDef):
        """Extract the model's names and arguments.
        """
        self.model.graph["name"] = node.name

        fn_args = node.args.args
        fn_defaults = node.args.defaults

        arguments = []
        for arg in fn_args:
            arguments.append(arg.arg)

        defaults = [None] * len(arguments)
        for i, default in enumerate(fn_defaults):
            defaults[-1 - i] = default.value

        for argument, default in zip(arguments, defaults):
            new_node = Argument()
            new_node.name = argument
            new_node.default = default
            self.model.add_node(argument, content=new_node)

    def visit_Arguments(self, node):
        arguments = []
        for arg in node.value.args:
            if isinstance(arg, ast.Name):
                arguments.append(arg.id)
            elif isinstance(arg, ast.Constant):
                arguments.append(arg.value)
            else:
                raise SyntaxError(
                    "Expected a random variable of a constant to initialize distribution, got {} instead.\n"
                    "Maybe you are trying to initialize a distribution directly, or call a function inside the "
                    "distribution initialization. While this would be a perfectly legitimate move, it is currently "
                    "not supported in mcmx. Use an intermediate variable instead: \n\n"
                    "Do not do `x @ Normal(Normal(0, 1), 1)` or `x @ Normal(my_function(10), 1)`, instead do "
                    " `y @ Normal(0, 1) & x @ Normal(y, 1)` and `y = my_function(10) & x @ Normal(y, 1)`".format(
                        astor.code_gen.to_source(arg)
                    )
                )
        return arguments

    # pass

    def visit_Return(self, node):
        if isinstance(node.value, ast.Tuple):
            for var in node.value.elts:
                if isinstance(var, ast.Name):
                    self.returned.append(var.id)
        elif isinstance(node.value, ast.Name):
            self.returned.append(node.value.id)
        else:
            raise SyntaxError(
                "Expected the generative model to return a (random) variable or a tuple of (random) variables, got {}".format(
                    node.value
                )
            )

    def visit_Deterministic(self, node: ast.Assign):
        assert len(node.targets) == 1
        target = node.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
            expression = None
            value = None
            if isinstance(node.value, ast.Constant):
                value = node.constant.value
            elif isinstance(node.value, ast.Call):
                expression, value = read_deterministic_expression(node.value)

            new_node = Var()
            new_node.name = name
            new_node.value = value
            new_node.expression = expression
        else:
            raise SyntaxError(
                "Require a name on the left-hand-side of a deterministic variable assignment, got {}".format(
                    target
                )
            )
        # args = self.visit_Arguments(node)
        # node = graph.Deterministic(name=name, fn=None, args=[])
        # self.graph.add_node(node)

    def visit_RandAssign(self, node):
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
        self.randvars.append(name)
        self.distributions.append(node.right)

        # Get arguments

        # self.visit_Arguments(node.right)
        # node = self.RandomVariable(name=name, distribution=Distribution, args=args)
        # self.graph.add_node(node)

        # Build random node
        # Add node

    def visit_Response(self, node):
        pass

    def visit_Expr(self, node):
        if isinstance(node.value, ast.BinOp):
            self.visit_BinOp(node.value)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self.visit_RandAssign(node)

    def visit_Assign(self, node):
        if isinstance(node.value, ast.Call):
            self.visit_Deterministic(node)
        elif isinstance(node.value, ast.Lambda):
            raise NotImplementedError
