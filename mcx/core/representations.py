"""Return the different representations of the probabilistic program.

Transformations are performed on the graphical model, which is then
compiled to the CST by the (universal) compiler.

"""
import copy
from collections import defaultdict
from functools import partial

import libcst as cst

import mcx.core.translation as t
from mcx.core.compiler import compile_graph
from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Op, Placeholder, SampleOp

__all__ = [
    "logpdf",
    "logpdf_contributions",
    "sample",
    "sample_joint",
    "sample_posterior_predictive",
]

# -------------------------------------------------------
#                    == LOGPDF ==
# --------------------------------------------------------


def logpdf(model):
    """Returns a function that compute the model's logpdf."""
    graph = copy.deepcopy(model.graph)
    graph = _logpdf_core(graph)

    # no node besides the logpdf is returned
    for node in graph.nodes():
        if isinstance(node, Op):
            node.is_returned = False

    # Create a new 'logpdf' node that is the sum of the individual variables'
    # contributions.
    def to_sum_of_logpdf(*args):
        def add(left, right):
            return cst.BinaryOperation(left, cst.Add(), right)

        if len(args) == 1:
            return t.name(args[0].value)
        elif len(args) == 2:
            left = cst.Name(args[0].value)
            right = cst.Name(args[1].value)
            return add(left, right)

        args = list(args)
        right = args.pop()
        left = args.pop()
        expr = add(left, right)
        for arg in args:
            right = args.pop()
            expr = add(expr, right)

        return expr

    logpdf_contribs = [node for node in graph if isinstance(node, SampleOp)]
    sum_node = Op(to_sum_of_logpdf, graph.name, "logpdf", is_returned=True)
    graph.add(sum_node, *logpdf_contribs)

    return compile_graph(graph, model.namespace, f"{graph.name}_logpdf")


def logpdf_contributions(model):
    """Return the variables' individual constributions to the logpdf.

    The function returns a dictionary {'var_name': logpdf_contribution}. When
    there are several scopes it returns a nested dictionary {'scope':
    {'var_name': logpdf_contribution}} to avoid name conflicts.

    We cheat a little here: the function that returns the ast takes the contrib
    nodes as arguments, but these are not used: the content of the function is
    fully determined before adding the node to the graph. We do not have a
    choice because it is currently impossible to pass context (variable name
    and scope name) at compilation.

    """
    graph = copy.deepcopy(model.graph)
    graph = _logpdf_core(graph)

    # no node besides the logpdf is returned
    for node in graph.nodes():
        if isinstance(node, Op):
            node.is_returned = False

    # add a new node, a dictionary that contains the contribution of each
    # variable to the log-probability.
    logpdf_contribs = [node for node in graph if isinstance(node, SampleOp)]

    def to_dictionary_of_contributions(*_):
        scopes = [contrib.scope for contrib in logpdf_contribs]
        contrib_names = [contrib.name for contrib in logpdf_contribs]
        var_names = [
            name.replace(f"logpdf_{scope}_", "")
            for name, scope in zip(contrib_names, scopes)
        ]

        scoped = defaultdict(dict)
        for scope, var_name, contrib_name in zip(scopes, var_names, contrib_names):
            scoped[scope][var_name] = contrib_name

        # if there is only one scope (99% of models) we return a flat dictionary
        if len(set(scopes)) == 1:
            scope = scopes[0]
            return t.dict(
                {
                    cst.SimpleString(f"'{var_name}'"): cst.Name(contrib_name)
                    for var_name, contrib_name in scoped[scope].items()
                }
            )

        # Otherwise we return a nested dictionary where the first level is
        # the scope, and then the variables.
        return t.dict(
            {
                cst.SimpleString(f"'{scope}'"): t.dict(
                    {
                        cst.SimpleString(f"'{var_name}'"): cst.Name(contrib_name)
                        for var_name, contrib_name in scoped[scope].items()
                    }
                )
                for scope in scoped.keys()
            }
        )

    tuple_node = Op(
        to_dictionary_of_contributions,
        graph.name,
        "logpdf_contributions",
        is_returned=True,
    )
    graph.add(tuple_node, *logpdf_contribs)

    return compile_graph(graph, model.namespace, f"{graph.name}_logpdf_contribs")


def _logpdf_core(graph: GraphicalModel):
    """Transform the SampleOps to statements that compute the logpdf associated
    with the variables' values.
    """
    placeholders = []
    sample = []

    def sample_to_logpdf(to_ast, *args, **kwargs):
        name = kwargs.pop("var_name")
        return t.call(
            cst.Attribute(to_ast(*args, **kwargs), cst.Name("logpdf_sum")), name
        )

    def placeholder_to_param(name: str):
        return t.param(name)

    for node in reversed(list(graph.nodes())):
        if not isinstance(node, SampleOp):
            continue

        # Create a new placeholder node with the random variable's name
        rv_name = node.name
        name_node = Placeholder(
            rv_name, partial(placeholder_to_param, rv_name), rv=True
        )
        placeholders.append(name_node)

        # Update the nodes
        node.to_ast = partial(sample_to_logpdf, node.to_ast)
        node.name = f"logpdf_{node.scope}_{node.name}"

        # The random variables now must be placeholder nodes pointing to
        # the logpdfs
        sample.append(node)

    # Add the placeholders to the graph
    for name_node, node in zip(placeholders, sample):
        graph.add_node(name_node)
        graph.add_edge(name_node, node, type="kwargs", key=["var_name"])

        # remove edges from the former SampleOp and replace by new placeholder
        to_remove = []
        for e in graph.out_edges(node):
            data = graph.get_edge_data(*e)
            to_remove.append(e)
            graph.add_edge(name_node, e[1], **data)

        for e in to_remove:
            graph.remove_edge(*e)

    return graph


# -------------------------------------------------------
#                   == PRIOR SAMPLING ==
# --------------------------------------------------------


def sample(model):
    """Sample from the predictive model."""
    graph = copy.deepcopy(model.graph)
    graph = _sampler_core(graph)
    return compile_graph(graph, model.namespace, f"{graph.name}_sample")


def sample_joint(model):
    """Obtain forward samples from the joint distribution defined by the model."""
    graph = copy.deepcopy(model.graph)
    namespace = model.namespace

    random_variables = graph.random_variables

    def to_ast(*_):
        scopes = [rv.scope for rv in random_variables]
        names = [rv.name for rv in random_variables]

        scoped = defaultdict(dict)
        for scope, var_name, var in zip(scopes, names, random_variables):
            scoped[scope][var_name] = var

        # if there is only one scope (99% of models) we return a flat dictionary
        if len(set(scopes)) == 1:

            scope = scopes[0]
            return cst.Dict(
                [
                    cst.DictElement(
                        cst.SimpleString(f"'{var_name}'"),
                        cst.Name(var.name),
                    )
                    for var_name, var in scoped[scope].items()
                ]
            )

        # Otherwise we return a nested dictionary where the first level is
        # the scope, and then the variables.
        return cst.Dict(
            [
                cst.DictElement(
                    cst.SimpleString(f"'{scope}'"),
                    cst.Dict(
                        [
                            cst.DictElement(
                                cst.SimpleString(f"'{var_name}'"),
                                cst.Name(var.name),
                            )
                            for var_name, var in scoped[scope].items()
                        ]
                    ),
                )
                for scope in scoped.keys()
            ]
        )

    # no node is returned anymore
    for node in graph.nodes():
        if isinstance(node, Op):
            node.is_returned = False

    tuple_node = Op(to_ast, graph.name, "forward_samples", is_returned=True)

    graph = _sampler_core(graph)
    graph.add(tuple_node, *random_variables)
    return compile_graph(graph, namespace, f"{graph.name}_sample_forward")


def _sampler_core(graph: GraphicalModel):
    """Transform the SampleOps to statements that compute the logpdf associated
    with the variables' values.
    """

    rng_node = Placeholder("rng_key", lambda: cst.Param(name=cst.Name(value="rng_key")))

    # Update the SampleOps to return a sample from the distribution
    def to_sampler(to_ast, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Attribute(value=to_ast(*args, **kwargs), attr=cst.Name("sample")),
            args=[cst.Arg(value=rng_key)],
        )

    random_variables = []
    for node in reversed(list(graph.nodes())):
        if not isinstance(node, SampleOp):
            continue
        node.to_ast = partial(to_sampler, node.to_ast)
        random_variables.append(node)

    # Add the placeholders to the graph
    graph.add(rng_node)
    for var in random_variables:
        graph.add_edge(rng_node, var, type="kwargs", key=["rng_key"])

    return graph


# -------------------------------------------------------
#                 == POSTERIOR SAMPLING ==
# --------------------------------------------------------


def sample_posterior_predictive(model, node_names):
    """Sample from the posterior predictive distribution.

    def linear_regression(X, lmbda=1.):
        scale <~ Exponential(lmbda)
        coef <~ Normal(jnp.zeros(X.shape[0]), 1)
        y = jnp.dot(X, coef)
        pred <~ Normal(y, scale)
        return pred

    def linear_regression_pred(rng_key, X, **trace, lambda=1.):
        scale = jax.random.choice(rng_key, scale)
        coef = jax.random.choice(rng_key, coef)
        y = jnp.dot(X, coef)
        pred = Normal(y, scale).sample(rng_key)
        return pred
    """
    graph = copy.deepcopy(model.graph)

    nodes = [graph.find_node(name) for name in node_names]

    # first remove all incoming edges
    to_remove = []
    for e in graph.in_edges(nodes):
        to_remove.append(e)

    for edge in to_remove:
        graph.remove_edge(*edge)

    # Add a rng placeholder
    rng_node = Placeholder("rng_key", lambda: cst.Param(name=cst.Name(value="rng_key")))
    graph.add_node(rng_node)

    # Choose a sample id at random
    def choice_ast(rng_key):
        return cst.Call(
            func=cst.Attribute(
                value=cst.Attribute(cst.Name("jax"), cst.Name("random")),
                attr=cst.Name("choice"),
            ),
            args=[
                cst.Arg(rng_key),
                cst.Arg(
                    cst.Subscript(
                        cst.Attribute(cst.Name(nodes[0].name), cst.Name("shape")),
                        [cst.SubscriptElement(cst.Index(cst.Integer("0")))],
                    )
                ),
            ],
        )

    choice_node = Op(choice_ast, graph.name, "idx")
    graph.add(choice_node, rng_node)

    # Every node is replaced by a placeholder for the posterior samples
    # and a sampling function.
    for node in reversed(nodes):
        rv_name = node.name

        # Add the placeholder
        placeholder = Placeholder(
            rv_name, partial(lambda name: t.param(name), rv_name), rv=True
        )
        graph.add_node(placeholder)

        def choose_ast(placeholder, idx):
            return cst.Subscript(placeholder, [cst.SubscriptElement(cst.Index(idx))])

        chosen_sample = Op(choose_ast, graph.name, rv_name + "_sample")
        graph.add(chosen_sample, placeholder, choice_node)

        original_edges = []
        for e in graph.out_edges(node):
            data = graph.get_edge_data(*e)
            original_edges.append(e)
            graph.add_edge(chosen_sample, e[1], **data)

        for e in original_edges:
            graph.remove_edge(*e)

        graph.remove_node(node)

    # recursively remove every node that has no outgoing edge and is not
    # returned
    graph = remove_dangling_nodes(graph)

    # replace SampleOps by sampling instruction
    def to_sampler(to_ast, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Attribute(value=to_ast(*args, **kwargs), attr=cst.Name("sample")),
            args=[cst.Arg(value=rng_key)],
        )

    random_variables = []
    for node in reversed(list(graph.nodes())):
        if not isinstance(node, SampleOp):
            continue
        node.to_ast = partial(to_sampler, node.to_ast)
        random_variables.append(node)

    # Add the placeholders to the graph
    for var in random_variables:
        graph.add_edge(rng_node, var, type="kwargs", key=["rng_key"])

    return compile_graph(
        graph, model.namespace, f"{graph.name}_sample_posterior_predictive"
    )


def is_dangling(graph, node):
    if len(graph.out_edges(node)) != 0:
        return False
    elif isinstance(node, Op) and node.is_returned:
        return False
    elif isinstance(node, Placeholder):
        return False
    return True


def remove_dangling_nodes(graph) -> GraphicalModel:
    dangling_nodes = [node for node in graph.nodes() if is_dangling(graph, node)]
    if dangling_nodes:
        for n in dangling_nodes:
            graph.remove_node(n)
        return remove_dangling_nodes(graph)
    else:
        return graph
