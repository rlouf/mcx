from functools import partial
from typing import Dict

import libcst as cst
import networkx as nx

from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Op, Placeholder, Constant, SampleOp


def logpdf(graph: GraphicalModel, namespace: Dict):

    def to_logpdf(to_ast, *args, **kwargs):
        return cst.Call(
            func=cst.Attribute(value=to_ast(*args), attr=cst.Name("logpdf")),
            args=[cst.Arg(value=cst.Name(value=kwargs['var_name']))],
        )

    placeholders = []
    sample = []
    for node in graph.nodes():
        if not isinstance(node, SampleOp):
            continue

        logpdf_ast = partial(to_logpdf, node.to_ast)
        scope = "unefined"
        new_name = f"logpdf_{scope}_{node.name}"

        # Update the node
        node.to_ast = logpdf_ast
        node.name = new_name

        # names become placeholder nodes
        name_node = Placeholder(node.name, lambda: cst.Name(value=node.name))
        placeholders.append(name_node)
        sample.append(node)

    for name_node, node in zip(placeholders, sample):
        graph.add_node(name_node)
        graph.add_edge(name_node, node, type="kwarg")

    compile_cst(graph, namespace)


def compile(graph, namespace):
    return compile_cst(graph, namespace)

def compile_cst(graph: GraphicalModel, namespace):
    """Compile MCX's graph into a python function.

    TODO
    - support function kwargs
    - support kwargs in ops
    """

    args = []
    stmts = []
    returns = []
    for node in nx.topological_sort(graph):

        if isinstance(node, Placeholder):
            arg = node.to_ast()
            args.append(arg)

        if isinstance(node, Constant):
            if node.name is not None:
                stmts.append(node.to_ast())

        if isinstance(node, Op):
            if node.name is not None:
                stmts.append(
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(value=node.name))],
                        value=compile_op(node, graph),
                    )
                )

        # if a named node has no successor it has to be returned.  The
        # following only works if 'successors' does not return 'None' (which it
        # shouldn't).
        if next(graph.successors(node), None) is None:
            returns.append(cst.Return(value=cst.Name(value=node.name)))

    # Build function using the previously compiled nodes
    args.reverse()
    ast_fn = cst.Module(
        body=[
            cst.FunctionDef(
                name=cst.Name(value=graph.name),
                params=cst.Parameters(params=args),
                body=cst.IndentedBlock(
                    body=[
                        cst.SimpleStatementLine(body=[stmt]) for stmt in stmts + returns
                    ]
                ),
            )
        ]
    )

    code = ast_fn.code
    print(code)
    exec(code, namespace)
    fn = namespace[graph.name]

    return fn


def compile_op(node: Op, graph: GraphicalModel):
    op_args = []
    op_kwargs = {}
    for predecessor in graph.predecessors(node):
        if graph[predecessor][node] == {}:
            pass
        if predecessor.name is not None:
            pred_ast = cst.Name(value=predecessor.name)
        else:
            pred_ast = compile_op(predecessor, graph)

        if graph[predecessor][node]["type"] == "arg":
            op_args.append(pred_ast)
        else:
            op_kwargs[graph[predecessor][node]["key"][0]] = pred_ast

    return node.to_ast(*op_args, **op_kwargs)
