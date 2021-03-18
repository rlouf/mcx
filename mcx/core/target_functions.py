"""Return the different representations of the probabilistic program.

Transformations are performed on the graphical model, which is then
compiled to the CST by the (universal) compiler.

"""
import copy
from collections import defaultdict
from functools import partial

import libcst as cst

from mcx.core.compiler import compile_graph
from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Op, Placeholder, SampleModelOp, SampleOp

__all__ = [
    "logpdf",
    "logpdf_contributions",
    "sample_predictive",
    "sample_joint",
    "sample_posterior_predictive",
]

# --------------------------------------------------------------------
#                            == LOGPDF ==
# --------------------------------------------------------------------


def logpdf(model):
    """Returns a function that computes the log-probability."""
    graph = copy.deepcopy(model.graph)
    graph = _logpdf_core(graph)

    # Create a new `logpdf` node that is the sum of the contributions of each variable.
    def to_sum_of_logpdf(*args):
        def add(left, right):
            return cst.BinaryOperation(left, cst.Add(), right)

        args = list(args)
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            left = args[0]
            right = args[1]
            return add(left, right)

        right = args.pop()
        left = args.pop()
        expr = add(left, right)
        for arg in args:
            expr = add(expr, arg)

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

    # add a new node, a dictionary that contains the contribution of each
    # variable to the log-probability.
    logpdf_contribs = [node for node in graph if isinstance(node, SampleOp)]

    scopes = set()
    scope_map = defaultdict(dict)
    for contrib in logpdf_contribs:
        var_name = (contrib.name).replace(f"logpdf_{contrib.scope}_", "")
        scope_map[contrib.scope][var_name] = contrib.name
        scopes.add(contrib.scope)

    def to_dictionary_of_contributions(*_):

        # if there is only one scope we return a flat dictionary {'var': logpdf_var}
        num_scopes = len(scopes)
        if num_scopes == 1:
            scope = scopes.pop()
            return cst.Dict(
                [
                    cst.DictElement(
                        cst.SimpleString(f"'{var_name}'"), cst.Name(contrib_name)
                    )
                    for var_name, contrib_name in scope_map[scope].items()
                ]
            )

        # Otherwise we return a nested dictionary where the first level is
        # the scope, and then the variables {'model': {}, 'submodel': {}}
        return cst.Dict(
            [
                cst.DictElement(
                    cst.SimpleString(f"'{scope}'"),
                    cst.Dict(
                        [
                            cst.DictElement(
                                cst.SimpleString(f"'{var_name}'"),
                                cst.Name(contrib_name),
                            )
                            for var_name, contrib_name in scope_map[scope].items()
                        ]
                    ),
                )
                for scope in scopes
            ]
        )

    dict_node = Op(
        to_dictionary_of_contributions,
        graph.name,
        "logpdf_contributions",
        is_returned=True,
    )
    graph.add(dict_node, *logpdf_contribs)

    return compile_graph(graph, model.namespace, f"{graph.name}_logpdf_contribs")


def _logpdf_core(graph: GraphicalModel):
    """Transform the SampleOps to statements that compute the logpdf associated
    with the variables' values.
    """
    placeholders = []
    logpdf_nodes = []

    def sampleop_to_logpdf(cst_generator, *args, **kwargs):
        name = kwargs.pop("var_name")
        return cst.Call(
            cst.Attribute(cst_generator(*args, **kwargs), cst.Name("logpdf_sum")),
            [cst.Arg(name)],
        )

    def samplemodelop_to_logpdf(model_name, *args, **kwargs):
        name = kwargs.pop("var_name")
        return cst.Call(
            cst.Attribute(cst.Name(model_name), cst.Name("logpdf")),
            list(args) + [cst.Arg(name, star="**")],
        )

    def placeholder_to_param(name: str):
        return cst.Param(cst.Name(name))

    for node in graph.random_variables:
        if not isinstance(node, SampleModelOp):
            continue

        rv_name = node.name
        returned_var_name = node.graph.returned_variables[0].name

        def sample_index(rv, returned_var, *_):
            return cst.Subscript(
                cst.Name(rv),
                [cst.SubscriptElement(cst.SimpleString(f"'{returned_var}'"))],
            )

        chosen_sample = Op(
            partial(sample_index, rv_name, returned_var_name),
            graph.name,
            f"{rv_name}_value",
        )

        original_edges = []
        data = []
        out_nodes = []
        for e in graph.out_edges(node):
            datum = graph.get_edge_data(*e)
            data.append(datum)
            original_edges.append(e)
            out_nodes.append(e[1])

        for e in original_edges:
            graph.remove_edge(*e)

        graph.add(chosen_sample, node)
        for e, d in zip(out_nodes, data):
            graph.add_edge(chosen_sample, e, **d)

    # We need to loop through the nodes in reverse order because of the compilation
    # quirk which makes it that nodes added first to the graph appear first in the
    # functions arguments. This should be taken care of properly before merging.
    for node in reversed(list(graph.random_variables)):

        # Create a new placeholder node with the random variable's name.
        # It represents the value that will be passed to the logpdf.
        name = node.name
        rv_placeholder = Placeholder(
            partial(placeholder_to_param, name), name, is_random_variable=True
        )
        placeholders.append(rv_placeholder)

        # Transform the SampleOps from `a <~ Normal(0, 1)` into
        # `lopdf_a = Normal(0, 1).logpdf_sum(a)`
        if isinstance(node, SampleModelOp):
            node.cst_generator = partial(samplemodelop_to_logpdf, node.model_name)
        else:
            node.cst_generator = partial(sampleop_to_logpdf, node.cst_generator)
        node.name = f"logpdf_{node.scope}_{node.name}"
        logpdf_nodes.append(node)

    for placeholder, node in zip(placeholders, logpdf_nodes):
        # Add the placeholder to the graph and link it to the expression that
        # computes the logpdf. So far the expression looks like:
        #
        #    >>> logpdf_a = Normal(0, 1).logpdf_sum(_)
        #
        # `a` is the placeholder and will appear into the arguments of
        # the function. Below we assign it to `_`.
        graph.add_node(placeholder)
        graph.add_edge(placeholder, node, type="kwargs", key=["var_name"])

        # Remove edges from the former SampleOp and replace by new placeholder
        # For instance, assume that part of our model is:
        #
        #     >>> a <~ Normal(0, 1)
        #     >>> x = jnp.log(a)
        #
        # Transformed to a logpdf this would look like:
        #
        #     >>> logpdf_a = Normal(0, 1).logpdf_sum(a)
        #     >>> x = jnp.log(a)
        #
        # Where a is now a placeholder, passed as an argument. The following
        # code links this placeholder to the expression `jnp.log(a)` and removes
        # the edge from `a <~ Normal(0, 1)`.
        #
        # We cannot remove edges while iterating over the graph, hence the two-step
        # process.
        # to_remove = []
        successors = list(graph.successors(node))
        for s in successors:
            edge_data = graph.get_edge_data(node, s)
            graph.add_edge(placeholder, s, **edge_data)

        for s in successors:
            graph.remove_edge(node, s)

    # The original MCX model may return one or many variables. None of
    # these variables should be returned, so we turn the `is_returned` flag
    # to `False`.
    for node in graph.nodes():
        if isinstance(node, Op):
            node.is_returned = False

    return graph


# -------------------------------------------------------
#                   == PRIOR SAMPLING ==
# --------------------------------------------------------


def sample_predictive(model):
    """Sample from the model's predictive distribution."""
    graph = copy.deepcopy(model.graph)

    rng_node = Placeholder(lambda: cst.Param(cst.Name(value="rng_key")), "rng_key")

    # Update the SampleOps to return a sample from the distribution so that
    # `a <~ Normal(0, 1)` becomes `a = Normal(0, 1).sample(rng_key)`.
    def distribution_to_sampler(cst_generator, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Attribute(cst_generator(*args, **kwargs), cst.Name("sample")),
            args=[cst.Arg(value=rng_key)],
        )

    def model_to_sampler(model_name, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Name(value=model_name), args=[cst.Arg(value=rng_key)] + list(args)
        )

    random_variables = []
    for node in reversed(list(graph.random_variables)):
        if isinstance(node, SampleModelOp):
            node.cst_generator = partial(model_to_sampler, node.model_name)
        else:
            node.cst_generator = partial(distribution_to_sampler, node.cst_generator)
        random_variables.append(node)

    # Link the `rng_key` placeholder to the sampling expressions
    graph.add(rng_node)
    for var in random_variables:
        graph.add_edge(rng_node, var, type="kwargs", key=["rng_key"])

    return compile_graph(graph, model.namespace, f"{graph.name}_sample")


def sample_joint(model):
    """Obtain forward samples from the joint distribution defined by the model."""
    graph = copy.deepcopy(model.graph)
    namespace = model.namespace

    def to_dictionary_of_samples(random_variables, *_):
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

    rng_node = Placeholder(lambda: cst.Param(cst.Name(value="rng_key")), "rng_key")

    # Update the SampleOps to return a sample from the distribution so that
    # `a <~ Normal(0, 1)` becomes `a = Normal(0, 1).sample(rng_key)`.
    def distribution_to_sampler(cst_generator, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Attribute(cst_generator(*args, **kwargs), cst.Name("sample")),
            args=[cst.Arg(value=rng_key)],
        )

    def model_to_sampler(model_name, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Attribute(cst.Name(value=model_name), cst.Name("sample")),
            args=[cst.Arg(value=rng_key)] + list(args),
        )

    random_variables = []
    for node in reversed(list(graph.random_variables)):
        if isinstance(node, SampleModelOp):
            node.cst_generator = partial(model_to_sampler, node.model_name)
        else:
            node.cst_generator = partial(distribution_to_sampler, node.cst_generator)
        random_variables.append(node)

    # Link the `rng_key` placeholder to the sampling expressions
    graph.add(rng_node)
    for var in random_variables:
        graph.add_edge(rng_node, var, type="kwargs", key=["rng_key"])

    for node in graph.random_variables:
        if not isinstance(node, SampleModelOp):
            continue

        rv_name = node.name
        returned_var_name = node.graph.returned_variables[0].name

        def sample_index(rv, returned_var, *_):
            return cst.Subscript(
                cst.Name(rv),
                [cst.SubscriptElement(cst.SimpleString(f"'{returned_var}'"))],
            )

        chosen_sample = Op(
            partial(sample_index, rv_name, returned_var_name),
            graph.name,
            rv_name + "_value",
        )

        original_edges = []
        data = []
        out_nodes = []
        for e in graph.out_edges(node):
            datum = graph.get_edge_data(*e)
            data.append(datum)
            original_edges.append(e)
            out_nodes.append(e[1])

        for e in original_edges:
            graph.remove_edge(*e)

        graph.add(chosen_sample, node)
        for e, d in zip(out_nodes, data):
            graph.add_edge(chosen_sample, e, **d)

    tuple_node = Op(
        partial(to_dictionary_of_samples, graph.random_variables),
        graph.name,
        "forward_samples",
        is_returned=True,
    )
    graph.add(tuple_node, *graph.random_variables)

    return compile_graph(graph, namespace, f"{graph.name}_sample_forward")


# -------------------------------------------------------
#                 == POSTERIOR SAMPLING ==
# --------------------------------------------------------


def sample_posterior_predictive(model, node_names):
    """Sample from the posterior predictive distribution.

    Example
    -------

    We transform MCX models of the form:

        >>> def linear_regression(X, lmbda=1.):
        ...    scale <~ Exponential(lmbda)
        ...    coef <~ Normal(jnp.zeros(X.shape[0]), 1)
        ...    y = jnp.dot(X, coef)
        ...    pred <~ Normal(y, scale)
        ...    return pred

    into:

        >>> def linear_regression_pred(rng_key, scale, coef, X, lambda=1.):
        ...    idx = jax.random.choice(rng_key, scale.shape[0])
        ...    scale_sample = scale[idx]
        ...    coef_sample = coef[idx]
        ...    y = jnp.dot(X, coef_sample)
        ...    pred = Normal(y, scale_sample).sample(rng_key)
        ...    return pred

    """
    graph = copy.deepcopy(model.graph)
    nodes = [graph.find_node(name) for name in node_names]

    # We will need to pass a RNG key to the function to sample from
    # the distributions; we create a placeholder for this key.
    rng_node = Placeholder(lambda: cst.Param(cst.Name(value="rng_key")), "rng_key")
    graph.add_node(rng_node)

    # To take a sampler from the posterior distribution we first choose a sample id
    # at random `idx = mcx.jax.choice(rng_key, num_samples)`. We later index each
    # array of samples passed by this `idx`.
    def choice_ast(rng_key):
        return cst.Call(
            func=cst.Attribute(
                value=cst.Attribute(cst.Name("mcx"), cst.Name("jax")),
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

    # Remove all edges incoming to the nodes that are targetted
    # by the intervention.
    to_remove = []
    for e in graph.in_edges(nodes):
        to_remove.append(e)

    for edge in to_remove:
        graph.remove_edge(*edge)

    # Each SampleOp that is intervened on is replaced by a placeholder that is indexed
    # by the index of the sample being taken.
    for node in reversed(nodes):
        rv_name = node.name

        # Add the placeholder
        placeholder = Placeholder(
            partial(lambda name: cst.Param(cst.Name(name)), rv_name),
            rv_name,
            is_random_variable=True,
        )
        graph.add_node(placeholder)

        def sample_index(placeholder, idx):
            return cst.Subscript(placeholder, [cst.SubscriptElement(cst.Index(idx))])

        chosen_sample = Op(sample_index, graph.name, rv_name + "_sample")
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
    def to_sampler(cst_generator, *args, **kwargs):
        rng_key = kwargs.pop("rng_key")
        return cst.Call(
            func=cst.Attribute(
                value=cst_generator(*args, **kwargs), attr=cst.Name("sample")
            ),
            args=[cst.Arg(value=rng_key)],
        )

    random_variables = []
    for node in reversed(list(graph.nodes())):
        if not isinstance(node, SampleOp):
            continue
        node.cst_generator = partial(to_sampler, node.cst_generator)
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
