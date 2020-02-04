import ast
from types import FunctionType
from typing import Dict, List, Tuple, Union

import astor
import networkx as nx

import mcx
from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Argument, RandVar


def compile_to_logpdf(
    graph: GraphicalModel, namespace: Dict, jit: bool = False, grad: bool = False
) -> Tuple[FunctionType, List[str], str]:
    """Compile a graphical model into a log-probability density function.

    Arguments
    ---------
    model:
        A probabilistic graphical model.
    namespace:
        The names contained in the model definition's global scope.
    jit:
        Whether to JIT-compile the logpdf.
    grad:
        Whether to add reverse-mode differentiation.

    Returns
    -------
    logpdf:
        A function that returns the log probability of a model at one point the
        parameter space. Can optionally return the gradient and be
        JIT-compiled.
    var_names:
        The name of the random variables arguments of the logpdf function, in
        the order in which they appear.
    logpdf_source:
        A string containing the source code of the logpdf. Useful for inspection
        by the user.
    """
    fn_name = graph.name + "_logpdf"
    args = [
        node[1]["content"].to_logpdf()
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
    ] + [
        ast.arg(arg=node[1]["content"].name, annotation=None)
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], RandVar)
    ]

    body: List[Union[ast.Assign, ast.Constant, ast.Num, ast.Return]] = []
    body.append(
        ast.Assign(
            targets=[ast.Name(id="logpdf", ctx=ast.Store())],
            value=ast.Constant(value=0),
        )
    )
    ordered_nodes = [
        graph.nodes[node]["content"]
        for node in nx.topological_sort(graph)
        if not isinstance(graph.nodes[node]["content"], Argument)
    ]
    for node in ordered_nodes:
        body.append(node.to_logpdf())

    returned = ast.Return(value=ast.Name(id="logpdf", ctx=ast.Load()))
    body.append(returned)

    logpdf_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    args=args,
                    vararg=None,
                    kwarg=None,
                    posonlyargs=[],
                    kwonlyargs=[],
                    defaults=[],
                    kw_defaults=[],
                ),
                body=body,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    logpdf_ast = ast.fix_missing_locations(logpdf_ast)
    logpdf = compile(logpdf_ast, filename="<ast>", mode="exec")
    exec(logpdf, namespace)

    args = [arg.arg for arg in args]

    return namespace[fn_name], args, astor.code_gen.to_source(logpdf_ast)


def compile_to_sampler(graph, namespace, jit=False):
    """Compile the model in a function that generates prior samples from the
    model's random variables.

    (TODO): Check that the parameters used to initialize distributions respect
    the constraints. This cannot be done dynamically once the model is JIT
    compiled.

    Args:
        model: A probabilistic program definition.
        namespace: The model definition's global scope.
    Returns:
        sample_fn: A JIT compiled function that returns prior predictive
            samples from the model. The function's signature is of the form:
            `model_sampler(rng_key, *args, sample_shape=())`
    """
    fn_name = graph.name + "_sampler"
    args = (
        [ast.arg(arg="rng_key", annotation=None)]
        + [
            node[1]["content"].to_sampler()
            for node in graph.nodes(data=True)
            if isinstance(node[1]["content"], Argument)
        ]
        + [ast.arg(arg="sample_shape", annotation=None)]
    )

    defaults = [ast.Tuple(elts=[], ctx=ast.Load())]

    body = []
    ordered_nodes = [
        graph.nodes[node]["content"]
        for node in nx.topological_sort(graph)
        if not isinstance(graph.nodes[node]["content"], Argument)
    ]
    for node in ordered_nodes:
        body.append(node.to_sampler(graph))

    returned = ast.Return(
        value=ast.Tuple(
            elts=[
                ast.Name(id=node.name, ctx=ast.Load())
                for node in ordered_nodes
                if not isinstance(node, mcx.core.graph.Var)
            ],
            ctx=ast.Load(),
        )
    )
    body.append(returned)

    sampler_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    args=args,
                    vararg=None,
                    kwarg=None,
                    posonlyargs=[],
                    kwonlyargs=[],
                    defaults=defaults,
                    kw_defaults=[],
                ),
                body=body,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    sampler_ast = ast.fix_missing_locations(sampler_ast)

    sampler = compile(sampler_ast, filename="<ast>", mode="exec")
    exec(sampler, namespace)

    args = [arg.arg for arg in args]

    return namespace[fn_name], args, astor.code_gen.to_source(sampler_ast)


def compile_to_forward_sampler(graph, namespace, jit=False):
    """Compile the model in a function that generates prior samples from the
    model's generated variables.

    (TODO): Check that the parameters used to initialize distributions respect
    the constraints. This cannot be done dynamically once the model is JIT
    compiled.

    Args:
        model: A probabilistic program definition.
        namespace: The model definition's global scope.
    Returns:
        sample_fn: A JIT compiled function that returns prior predictive
            samples from the model. The function's signature is of the form:
            `model_sampler(rng_key, *args, sample_shape=())`
    """
    fn_name = graph.name + "_forward_sampler"
    args = (
        [ast.arg(arg="rng_key", annotation=None)]
        + [
            node[1]["content"].to_sampler()
            for node in graph.nodes(data=True)
            if isinstance(node[1]["content"], Argument)
        ]
        + [ast.arg(arg="sample_shape", annotation=None)]
    )

    defaults = [ast.Tuple(elts=[], ctx=ast.Load())]

    body = []
    ordered_nodes = [
        graph.nodes[node]["content"]
        for node in nx.topological_sort(graph)
        if not isinstance(graph.nodes[node]["content"], Argument)
    ]
    for node in ordered_nodes:
        body.append(node.to_sampler(graph))

    returned_vars = [
        ast.Name(id=node.name, ctx=ast.Load())
        for node in ordered_nodes
        if not isinstance(node, mcx.core.graph.Var) and node.is_returned
    ]
    if len(returned_vars) == 1:
        returned = ast.Return(returned_vars[0])
    else:
        returned = ast.Return(value=ast.Tuple(elts=returned_vars, ctx=ast.Load(),))
    body.append(returned)

    sampler_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    args=args,
                    vararg=None,
                    kwarg=None,
                    posonlyargs=[],
                    kwonlyargs=[],
                    defaults=defaults,
                    kw_defaults=[],
                ),
                body=body,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    sampler_ast = ast.fix_missing_locations(sampler_ast)

    sampler = compile(sampler_ast, filename="<ast>", mode="exec")
    exec(sampler, namespace)

    args = [arg.arg for arg in args]

    return namespace[fn_name], args, astor.code_gen.to_source(sampler_ast)
