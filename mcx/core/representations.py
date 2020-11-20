"""Functions that transform probabilistic programs into their
various representations.

Transformations are performed on the graphical model, which is then
compiled to the CST.

"""
from collections import defaultdict
from functools import partial
from typing import Dict

import libcst as cst

from mcx.core.compiler import compile_graph
from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Op, Placeholder, SampleOp

# -------------------------------------------------------
#                    == LOGPDF ==
# --------------------------------------------------------


def logpdf(graph: GraphicalModel, namespace: Dict):
    """Returns a function that compute the model's logpdf."""

    # Create a new 'logpdf' node that is the sum of the individual variables'
    # contributions.
    def to_ast(*args):
        if len(args) == 1:
            return cst.Name(value=args[0].name)

        def add(left, right):
            return cst.BinaryOperation(left=left, operator=cst.Add(), right=right)

        args = list(args)
        right = args.pop()
        left = args.pop()
        expr = add(left, right)
        for arg in args:
            right = args.pop()
            expr = add(expr, right)

        return expr

    sum_node = Op(to_ast, graph.name, "logpdf")

    graph = _logpdf_core(graph, namespace)
    logpdf_contribs = [node for node in graph if isinstance(node, SampleOp)]
    graph.add(sum_node, *logpdf_contribs)

    return compile_graph(graph, namespace)


def logpdf_contributions(graph: GraphicalModel, namespace: Dict):
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
            return cst.Dict(
                [
                    cst.DictElement(
                        cst.Name(value=var_name), cst.Name(value=contrib_name)
                    )
                    for var_name, contrib_name in scoped[scope].items()
                ]
            )

        # Otherwise we return a nested dictionary where the first level is
        # the scope, and then the variables.
        return cst.Dict(
            [
                cst.DictElement(
                    cst.Name(value=scope),
                    cst.Dict(
                        [
                            cst.DictElement(
                                cst.Name(value=var_name), cst.Name(value=contrib_name)
                            )
                            for var_name, contrib_name in scoped[scope].items()
                        ]
                    ),
                )
                for scope in scoped.keys()
            ]
        )

    tuple_node = Op(to_ast, graph.name, "logpdf_contributions")

    graph = _logpdf_core(graph, namespace)
    graph.add(tuple_node, *logpdf_contribs)

    return compile_graph(graph, namespace)


def _logpdf_core(graph: GraphicalModel, namespace: Dict):
    """Transform the SampleOps to statements that compute the logpdf associated
    with the variables' values.
    """
    placeholders = []
    sample = []

    def to_logpdf(to_ast, *args, **kwargs):
        print(args, kwargs)
        name = kwargs.pop("var_name")
        return cst.Call(
            func=cst.Attribute(value=to_ast(*args, **kwargs), attr=cst.Name("logpdf")),
            args=[cst.Arg(value=name)],
        )

    def to_placeholder_ast(name: str):
        return cst.Param(name=cst.Name(value=name))

    for node in graph.nodes():
        if not isinstance(node, SampleOp):
            continue

        # Create a new placeholder node with the random variable's name
        rv_name = node.name
        name_node = Placeholder(rv_name, partial(to_placeholder_ast, rv_name), rv=True)
        placeholders.append(name_node)

        # Update the nodes
        node.to_ast = partial(to_logpdf, node.to_ast)
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
    # Here we need to change the function return
    # 1 - Tuple logpdf_1, logpdf_2, etc. for logpdf contribs
    # 2 - Sum of logpdfs otherwise

    return graph
