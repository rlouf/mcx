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
    namespace = model.namespace

    # Create a new 'logpdf' node that is the sum of the individual variables'
    # contributions.
    def to_ast(*args):
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

    # no node is returned anymore
    for node in graph.nodes():
        if isinstance(node, Op):
            node.is_returned = False

    sum_node = Op(to_ast, graph.name, "logpdf", is_returned=True)

    graph = _logpdf_core(graph)
    logpdf_contribs = [node for node in graph if isinstance(node, SampleOp)]
    graph.add(sum_node, *logpdf_contribs)

    return compile_graph(graph, namespace, f"{graph.name}_logpdf")


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
    namespace = model.namespace

    logpdf_contribs = [node for node in graph if isinstance(node, SampleOp)]

    def to_ast(*_):
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

    # no node is returned anymore
    for node in graph.nodes():
        if isinstance(node, Op):
            node.is_returned = False

    tuple_node = Op(to_ast, graph.name, "logpdf_contributions", is_returned=True)

    graph = _logpdf_core(graph)
    graph.add(tuple_node, *logpdf_contribs)

    return compile_graph(graph, namespace, f"{graph.name}_logpdf_contribs")


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


def sample_posterior_predictive(model):
    """Sample from the posterior predictive distribution.

    Posterior sampling consists in choosing one of the samples
    at random, replacing each SampleOp with its value and
    propagate forward.

    `sample_posterior_predictive` can be seen as a particular case
    of `sample` where all variables are being conditioned on. It would
    thus make sense to merge both function by getting a list of variables
    we are conditioning on as an input.

    """
    graph = copy.deepcopy(model.graph)

    # Add `rng_key`, `num_samples` as arguments and sample choice Op
    def choice_ast(rng_key, num_samples):
        return t.call(
            cst.Attribute(
                attr=cst.Name("jax"),
                value=cst.Attribute(attr=cst.Name("random"), value=cst.Name("choice")),
            ),
            [rng_key, num_samples],
        )

    rng_node = Placeholder("rng_key", lambda: cst.Param(name=cst.Name(value="rng_key")))
    num_samples_node = Placeholder(
        "num_samples", lambda: cst.Param(name=cst.Name(value="num_samples"))
    )
    choice_node = Op(choice_ast, graph.name, "sample_id")
    graph.add(choice_node, rng_node, num_samples_node)

    # Modify `SampleOp` so it returns the value of sample `sample_id`
    def to_slicing(name, sample_id):
        """ get the `sample_id`th sample, name[:sample_id] """
        return cst.Subscript(
            cst.Name(name),
            slice=[
                cst.SubscriptElement(
                    lower=None, upper=cst.Name(sample_id), first_colon=cst.Colon()
                )
            ],
        )

    def placeholder_to_param(name: str):
        return t.param(name)

    random_variables = []
    placeholders = []
    for node in reversed(list(graph.nodes())):
        if not isinstance(node, SampleOp):
            continue
        node.to_ast = partial(to_slicing, node.name)
        random_variables.append(node)

        # Create a new placeholder node with the random variable's name
        rv_name = node.name
        name_node = Placeholder(
            rv_name, partial(placeholder_to_param, rv_name), rv=True
        )
        placeholders.append(name_node)

    # Add the `rng_key` and random variables as placeholders to the graph
    graph.add(rng_node)
    for name_var, var in zip(placeholders, random_variables):
        graph.add_edge(var, choice_node)
        graph.add_node(name_var)
        graph.add_edge(name_node, node, type="kwargs", key=["var_name"])

        # remove edges from the former SampleOp and replace by new placeholder
        to_remove = []
        for e in graph.out_edges(node):
            data = graph.get_edge_data(*e)
            to_remove.append(e)
            graph.add_edge(name_node, e[1], **data)

        for e in to_remove:
            graph.remove_edge(*e)


    return compile_graph(graph, model.namespace, f"{graph.name}_sample_posterior_predictive")
