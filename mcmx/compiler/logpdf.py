import ast
import inspect
from typing import Callable, Dict, List, Tuple, Union, Optional

import astor
import jax

from .utils import read_object_name


def compile_to_logpdf(
    model: Callable, namespace: Dict, grad: bool = False
) -> Tuple[Callable, List[str]]:
    """Compiles a model expression into a log-probability density function that
    can be evaluated by a sampler.

    Args:
        model: A probabilistic program definition.
        namespace: The model definition's global scope.
        grad: Whether to add reverse-mode differentiation.

    Returns
        logpdf: An JIT-compiled function that returns the log probability of
            a model at one point the parameter space.
        var_names: The name of the random variables arguments of the logpdf function, in
            the order in which they appear.
    """
    source = inspect.getsource(model)
    tree = ast.parse(source)

    compiler = LogpdfCompiler()
    tree = compiler.visit(tree)
    var_names = compiler.var_names
    print(astor.code_gen.to_source(tree))

    logpdf = compile(tree, filename="<ast>", mode="exec")
    exec(logpdf, namespace)
    logpdf = namespace[compiler.fn_name]
    if grad:
        logpdf = jax.grad(logpdf)
    logpdf = jax.jit(logpdf)
    return logpdf, var_names


class LogpdfCompiler(ast.NodeTransformer):
    def __init__(self) -> None:
        super(LogpdfCompiler, self).__init__()
        self.model_arguments: List[str] = []
        self.var_names: List[str] = []
        self.logpdf_names: List[str] = []
        self.fn_name: str = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.model_arguments = [argument.arg for argument in node.args.args]

        # We recursively visit children nodes before changing the function's
        # signature because we need to know the names of the random variables.
        self.generic_visit(node)

        new_node = node
        new_node.name = node.name + "_logpdf"
        new_node.args.args = new_logpdf_args(self.var_names)

        self.fn_name = new_node.name

        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return node

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        """Visit the ast.Expr nodes.

        If the expression is a random variable assignement `x @ ...` we change the
        expression to bind the variable to the value of the logpdf of the distribution
        at the position passed to the function. We keep the original node otherwise.

        Args:
            node:
                An `ast.Expr` node.

        Returns:
            A sampling assignment if the `ast.Expr` node correponds to a random variable assignment,
            the original node otherwise.

        Raises:
            ValueError:
                If the left-hand-side of the `@` symbol is not a variable name.
            ValueError:
                If the right-hand-side of the `@` symbol is not a function call.
        """
        if isinstance(node.value, ast.BinOp):
            if isinstance(node.value.op, ast.MatMult):

                # Check that we have a variable name on the let-hand side of `@`
                if isinstance(node.value.left, ast.Name):
                    var_name = node.value.left.id
                    self.var_names.append(var_name)
                else:
                    raise ValueError(
                        "Expected a name on the left of the random variable assignement, got {}",
                        astor.code_gen.to_source(node.value.left),
                    )

                if isinstance(node.value.right, ast.Call):
                    distribution = read_object_name(node.value.right.func)

                    arguments = node.value.right.args
                    for arg in arguments:
                        if not (isinstance(arg, ast.Name) or isinstance(arg, ast.Constant)):
                            raise ValueError(
                                "Expected a random variable of a constant to initialize {}'s distribution, got {} instead.\n"
                                "Maybe you are trying to initialize a distribution directly, or call a function inside the "
                                "distribution initialization. While this would be a perfectly legitimate move, it is currently "
                                "not supported in mcmx. Use an intermediate variable instead: \n\n"
                                "Do not do `x @ Normal(Normal(0, 1), 1)` or `x @ Normal(my_function(10), 1)`, instead do "
                                " `y @ Normal(0, 1) & x @ Normal(y, 1)` and `y = my_function(10) & x @ Normal(y, 1)`".format(
                                    var_name, astor.code_gen.to_source(arg)
                                )
                            )

                    args = arguments_not_defined(arguments, self.var_names, self.model_arguments)
                    if args:
                        raise ValueError(
                            "The random variable `{}` assignment {} "
                            "references arguments {} that have not been previously defined in the model definition's local scope.\n"
                            "Maybe you made a typo, forgot a definition or used a variable defined outside "
                            "of the model's definition. In the later case, please move the variable's definition "
                            "within the function that specified the model".format(
                                var_name, astor.code_gen.to_source(node.value.right), ",".join(args)
                            )
                        )

                    new_node, logpdf_name = new_logpdf_expression(var_name, distribution, arguments)
                    self.logpdf_names.append(logpdf_name)

                    ast.copy_location(new_node, node)
                    ast.fix_missing_locations(new_node)
                    return new_node
                else:
                    raise ValueError(
                        "Expected a distribution initialization on the right of the random variable assignement, got {}",
                        astor.code_gen.to_source(node.value.right),
                    )
        return node

    def visit_Return(self, node):
        new_node = new_return_statement(self.logpdf_names)
        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        return new_node


def new_logpdf_expression(
    var_name: str, distribution_name: str, arguments: List[Union[ast.expr, ast.Name, ast.Constant]]
):
    """Transform a random variable definition into a logpdf evaluation.

    Example:

        >>> x @ Normal(0, 1)

        is turned into

        >>> x_logprob = Normal(0, 1).logpdf(x)

    Args:
        var_name: The name of the random variable being defined.
        distribution_name: The full path to the distribution object.
        arguments: Arguments used to initialize the distibution.

    Returns:
        An expression that assigns the value of the distribution's logpdf at
        the current position to a new variable.
    """
    logpdf_name = var_name + "_logprob"
    node = ast.Assign(
        targets=[ast.Name(id=logpdf_name, ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id=distribution_name, ctx=ast.Load()),
                    args=arguments,
                    keywords=[],
                ),
                attr="logpdf",
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=var_name, ctx=ast.Load())],
            keywords=[],
        ),
        type_comment=None,
    )
    return node, logpdf_name


def new_return_statement(logprob_names: List[str]) -> ast.Return:
    """Creates a new statement to return the sum of the log probabilities
    computed in the function.

    Args
        logprob_names: The names of the variables that reference the computed
            log probabilities.
    """
    logpdf_vars = [ast.Name(id=name, ctx=ast.Load()) for name in logprob_names]
    node = ast.Return(
        value=recursive_sum_generation(logpdf_vars), ctx=ast.Load(), decorator_list=[],
    )
    return node


def new_logpdf_args(var_names: List[str]) -> List[ast.arg]:
    """Create the list of arguments for the model's log probability density
    function.
    """
    return [ast.arg(arg=name, annotation=None) for name in var_names]


def recursive_sum_generation(
    nodes: List[ast.Name], expr: Union[None, ast.BinOp] = None
) -> Optional[ast.BinOp]:
    """Recursively Generate the sum of variabes that reference the log
    probabilies.

    Args:
        nodes: A list containing the variables that reference the log probabilities.
        expr: An ast.BinOp node containing the current state of the sum.

    Returns
        The updated ast.BinOp node.
    """
    if not nodes:
        return expr
    if not expr:
        l, r = nodes.pop(), nodes.pop()
        return recursive_sum_generation(nodes, ast.BinOp(left=l, op=ast.Add(), right=r,),)

    node = nodes.pop()
    return recursive_sum_generation(nodes, ast.BinOp(left=expr, op=ast.Add(), right=node))


def arguments_not_defined(
    params: List[Union[ast.expr, ast.Name, ast.Constant]],
    model_vars: List[str],
    model_arguments: List[str],
) -> List[str]:
    """Checks that all arguments to the distribution's initialization are either constants
    or were passed as arguments to the model definition.

    Args:
        params: The parameters used to initialized the current random variable's distribution.
        model_vars: The random variables that have been visited so far in the traversal.

    Returns:
       A list that contains the names of the arguments that are not defined.
    """
    local_scope = model_vars + model_arguments
    args = []
    for p in params:
        if isinstance(p, ast.Name):
            if p.id not in local_scope:
                args.append(p.id)
    return args
