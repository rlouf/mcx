import ast
from types import FunctionType
from typing import Dict, List, NamedTuple, Union

import astor
import networkx as nx

from mcx.compiler.graph import GraphicalModel, Var
from mcx.compiler.nodes import Argument, RandVar


class Artifact(NamedTuple):
    compiled_fn: FunctionType
    args: List[str]
    fn_name: str
    fn_source: str


def compile_to_logpdf(graph: GraphicalModel, namespace: Dict) -> Artifact:
    """Compile a graphical model into a log-probability density function.

    Example
    -------

    Let us consider a simple linear regression example:

        >>> @mcx.model
        ... def linear_regression(x, lmbda=1.):
        ...     scale <~ Exponential(lmbda)
        ...     coeff <~ Normal(0, 1)
        ...     y = np.dot(x, coeff)
        ...     predictions <~ Normal(y, scale)
        ...     return predictions

    MCX parses this definition into a graphical model. This function compiles
    the graph in a python function that returns the values of the
    log-probability density function:

        >>> def linear_regression_logpdf(x, scale, coeffs, predictions, lmbda=1.):
        ...     logpdf = 0
        ...     logpdf += Exponential(lmbda).logpdf(scale)
        ...     logpdf += Normal(0, 1).logpdf(coeff)
        ...     y = np.dot(x, coeff)
        ...     logpdf += Normal(y, coeff).logpdf(predictions)
        ...     return logpdf

    The logpdf is then partially applied on the dataset {(x, prediction)} for
    inference.

    Of course it would impact the core


    Parameters
    ----------
    model:
        A probabilistic graphical model.
    namespace:
        The names contained in the model definition's global scope.

    Returns
    -------
    logpdf:
        A function that returns the log probability of a model at one point the
        parameter space.
    var_names:
        The name of the random variables arguments of the logpdf function, in
        the order in which they appear.
    logpdf_source:
        A string containing the source code of the logpdf. Useful for inspection
        by the user.
    """
    fn_name = graph.name + "_logpdf"

    #
    # ARGUMENTS
    #

    kwarg_nodes = [
        node[1]["content"]
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is not None
    ]

    # The (keyword) arguments of the model definition and random variables
    # are passed as arguments to the logpdf.
    model_kwargs = [kwarg.to_logpdf_iadd() for kwarg in kwarg_nodes]
    model_arguments = [
        node[1]["content"].to_logpdf_iadd()
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is None
    ]
    random_variables = [
        ast.arg(arg=node[1]["content"].name, annotation=None)
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], RandVar)
    ]

    logpdf_arguments = random_variables + model_arguments + model_kwargs

    # We propagate the kwargs' default values
    defaults = [kwarg.default_value for kwarg in kwarg_nodes]

    #
    # FUNCTION BODY
    # To write the function body, we traverse the graph in topological order
    # while incrementing the value of the logpdf.
    #

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
        body.append(node.to_logpdf_iadd())

    returned = ast.Return(value=ast.Name(id="logpdf", ctx=ast.Load()))
    body.append(returned)

    logpdf_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    args=logpdf_arguments,
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
    logpdf_ast = ast.fix_missing_locations(logpdf_ast)
    logpdf = compile(logpdf_ast, filename="<ast>", mode="exec")
    exec(logpdf, namespace)
    fn = namespace[fn_name]

    argument_names = [arg.arg for arg in logpdf_arguments]

    return Artifact(fn, argument_names, fn_name, astor.code_gen.to_source(logpdf_ast))


def compile_to_loglikelihoods(graph: GraphicalModel, namespace: Dict) -> Artifact:
    """

    Example
    -------

    Let us consider a simple linear regression example:

        >>> @mcx.model
        ... def linear_regression(x, lmbda=1.):
        ...     scale <~ Exponential(lmbda)
        ...     coeff <~ Normal(0, 1)
        ...     y = np.dot(x, coeff)
        ...     predictions <~ Normal(y, scale)
        ...     return predictions

    We can get the log-likelihood contribution of each parameter by doing:

        >>> def linear_regression_logpdf(x, scale, coeffs, predictions, lmbda=1.):
        ...     logpdf_scale = Exponential(lmbda).logpdf(scale)
        ...     logpdf_coeff = Normal(0, 1).logpdf(coeff)
        ...     y = np.dot(x, coeff)
        ...     logpdf_predictions = Normal(y, coeff).logpdf(predictions)
        ...     return np.array([logpdf_scale, logpdf_coeff, logpdf_predictions])
    """

    fn_name = graph.name + "_loglikelihoods"

    #
    # ARGUMENTS
    #

    kwarg_nodes = [
        node[1]["content"]
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is not None
    ]

    # The (keyword) arguments of the model definition and random variables
    # are passed as arguments to the logpdf.
    model_kwargs = [kwarg.to_logpdf() for kwarg in kwarg_nodes]
    model_arguments = [
        node[1]["content"].to_logpdf()
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is None
    ]
    random_variables = [
        ast.arg(arg=node[1]["content"].name, annotation=None)
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], RandVar)
    ]

    logpdf_arguments = random_variables + model_arguments + model_kwargs

    # We propagate the kwargs' default values
    defaults = [kwarg.default_value for kwarg in kwarg_nodes]

    #
    # FUNCTION BODY
    # To write the function body, we traverse the graph in topological order
    # while incrementing the value of the logpdf.
    #

    body: List[Union[ast.Assign, ast.Constant, ast.Num, ast.Return]] = []
    ordered_nodes = [
        graph.nodes[node]["content"]
        for node in nx.topological_sort(graph)
        if not isinstance(graph.nodes[node]["content"], Argument)
    ]
    # ordered_transformations = [
    # graph.nodes[node]["content"].name
    # for node in nx.topological_sort(graph)
    # if isinstance(graph.nodes[node]["content"], Transformation)
    # ]
    for node in ordered_nodes:
        body.append(node.to_logpdf())

    # Returned values
    #
    # In this situation we want to function to return both the individual
    # values of the log-likelihood as well as the values of the deterministic
    # variables.

    returned = ast.Return(
        value=ast.Dict(
            keys=[
                ast.Constant(value=f"{name.arg}", kind=None)
                for name in random_variables
            ],
            values=[
                ast.Name(id=f"logpdf_{name.arg}", ctx=ast.Load())
                for name in random_variables
            ],
        ),
        type_ignores=[],
    )
    body.append(returned)

    logpdf_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    args=logpdf_arguments,
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
    logpdf_ast = ast.fix_missing_locations(logpdf_ast)
    logpdf = compile(logpdf_ast, filename="<ast>", mode="exec")
    exec(logpdf, namespace)
    fn = namespace[fn_name]

    argument_names = [arg.arg for arg in logpdf_arguments]

    return Artifact(fn, argument_names, fn_name, astor.code_gen.to_source(logpdf_ast))


def compile_to_sampler(graph, namespace) -> Artifact:
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

    kwargs = [
        node[1]["content"]
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is not None
    ]

    args = (
        [ast.arg(arg="rng_key", annotation=None)]
        + [
            node[1]["content"].to_sampler()
            for node in graph.nodes(data=True)
            if isinstance(node[1]["content"], Argument)
            and node[1]["content"].default_value is None
        ]
        + [k.to_sampler() for k in kwargs]
    )

    defaults = [k.default_value for k in kwargs]

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
        if not isinstance(node, Var)
    ]

    returned = ast.Return(
        value=ast.Tuple(
            elts=returned_vars,
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
    fn = namespace[fn_name]

    returned_names = [arg.id for arg in returned_vars]

    return Artifact(fn, returned_names, fn_name, astor.code_gen.to_source(sampler_ast))


def compile_to_prior_sampler(graph, namespace, jit=False) -> Artifact:
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

    kwargs = [
        node[1]["content"]
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is not None
    ]

    args = (
        [ast.arg(arg="rng_key", annotation=None)]
        + [
            node[1]["content"].to_sampler()
            for node in graph.nodes(data=True)
            if isinstance(node[1]["content"], Argument)
            and node[1]["content"].default_value is None
        ]
        + [k.to_sampler() for k in kwargs]
    )

    defaults = [k.default_value for k in kwargs]

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
        if not isinstance(node, Var) and node.is_returned
    ]
    if len(returned_vars) == 1:
        returned = ast.Return(returned_vars[0])
    else:
        returned = ast.Return(
            value=ast.Tuple(
                elts=returned_vars,
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

    returned_names = [arg.id for arg in returned_vars]

    return Artifact(
        namespace[fn_name],
        returned_names,
        fn_name,
        astor.code_gen.to_source(sampler_ast),
    )


def compile_to_posterior_sampler(graph, namespace, jit=False) -> Artifact:
    """Compile the model in a function that generates one posterior predictive
    samples from the model's.

    Example
    -------

        >>> @mcx.model
        ... def linear_regression(x):
        ...     sigma <~ Exponential(1.)
        ...     weights <~ Normal(0, 1)
        ...     z = np.dot(X, weights)
        ...     y <~ Normal(z, sigma)
        ...     return y

    Should become:

        >>> @mcx.model
        ... def linear_regression(rng_key, x, sigma, weights):
        ...     z = np.dot(X, weights)
        ...     y = Normal(z, sigma).sample(rng_key)
        ...     return y

    Args:
        model: A probabilistic program definition.
        namespace: The model definition's global scope.
    Returns:
        sample_fn: A JIT compiled function that returns prior predictive
            samples from the model. The function's signature is of the form:
            `model_sampler(rng_key, *args, sample_shape=())`
    """
    fn_name = graph.name + "_posterior_sampler"

    args = [
        node[1]["content"].to_sampler()
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is None
    ]

    kwargs = [
        node[1]["content"]
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], Argument)
        and node[1]["content"].default_value is not None
    ]

    # Build the list of variables for which we will have posterior samples.
    # Essentially all random_variables except the returned ones.
    sampled_vars = [
        ast.arg(arg=node[1]["content"].name, annotation=None)
        for node in graph.nodes(data=True)
        if isinstance(node[1]["content"], RandVar)
        and node[1]["content"].is_returned is False
    ]

    fn_args = (
        [ast.arg(arg="rng_key", annotation=None)]
        + args
        + sampled_vars
        + [k.to_sampler() for k in kwargs]
    )

    fn_defaults = [k.default_value for k in kwargs]

    # Simplify the graph.
    #
    # Not all variables sampled are necessary for predictions. We thus
    # simplify the graph to avoid unnecessary computations. Note that we
    # still take these variables as arguments to ease the predictor's life.
    #
    # We proceed by removing every random variable that is not returned
    # from the graph and any deterministic computation in between.

    pruned_graph = graph.copy()
    for node in nx.topological_sort(graph):
        var = graph.nodes[node]["content"]
        if isinstance(var, RandVar) and not var.is_returned:
            pruned_graph.remove_node(node)

    connected_components = nx.algorithms.weakly_connected_components(pruned_graph)
    nodes_to_keep = []
    for component in connected_components:
        has_returned = sum(
            [pruned_graph.nodes[node]["content"].is_returned for node in component]
        )
        if has_returned:
            nodes_to_keep += [node for node in component]

    pruned_graph = nx.subgraph(pruned_graph, nodes_to_keep)

    # Now add the variables to the function's body
    body = []
    ordered_nodes = [
        pruned_graph.nodes[node]["content"]
        for node in nx.topological_sort(pruned_graph)
        if not isinstance(pruned_graph.nodes[node]["content"], Argument)
    ]
    for node in ordered_nodes:
        body.append(node.to_sampler(graph))

    returned_vars = [
        ast.Name(id=node.name, ctx=ast.Load())
        for node in ordered_nodes
        if not isinstance(node, Var) and node.is_returned
    ]
    if len(returned_vars) == 1:
        returned = ast.Return(returned_vars[0])
    else:
        returned = ast.Return(
            value=ast.Tuple(
                elts=returned_vars,
                ctx=ast.Load(),
            )
        )
    body.append(returned)
    sampler_ast = ast.Module(
        body=[
            ast.FunctionDef(
                name=fn_name,
                args=ast.arguments(
                    args=fn_args,
                    vararg=None,
                    kwarg=None,
                    posonlyargs=[],
                    kwonlyargs=[],
                    defaults=fn_defaults,
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

    returned_names = [arg.id for arg in returned_vars]

    return Artifact(
        namespace[fn_name],
        returned_names,
        fn_name,
        astor.code_gen.to_source(sampler_ast),
    )
