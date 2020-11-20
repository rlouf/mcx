import libcst as cst
import networkx as nx

from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Op, Placeholder, Constant


def compile_graph(graph: GraphicalModel, namespace):
    """Compile MCX's graph into a python (executable) function."""

    args = []
    stmts = []
    returns = []

    # Model arguments. Random variables (if logpdf) first then the model's arguments
    maybe_random_variables = [node for node in graph.nodes if isinstance(node, Placeholder) and node.rv]
    model_args = [node for node in graph.nodes if isinstance(node, Placeholder) and not node.rv]
    args = [placeholder.to_ast() for placeholder in maybe_random_variables] + [placeholder.to_ast() for placeholder in model_args]

    # Every statement in the function corresponds to either a constant definition or
    # a variable assignment. We use a topological sort to respect the
    # dependency order.
    for node in nx.topological_sort(graph):

        if node.name is None:
            continue

        if isinstance(node, Constant):
            stmt = cst.SimpleStatementLine(body=[node.to_ast()])
            stmts.append(stmt)

        if isinstance(node, Op):
            stmt = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[
                            cst.AssignTarget(target=cst.Name(value=node.name))
                        ],
                        value=compile_op(node, graph),
                    )
                ]
            )
            stmts.append(stmt)

        # We return nodes that do not have a successor in the graph. The
        # following only works if 'successors' does not return 'None' (which it
        # shouldn't).
        if next(graph.successors(node), None) is None:
            returns.append(cst.SimpleStatementLine(body=[cst.Return(value=cst.Name(value=node.name))]))

    # Assemble the function's CST using the previously translated nodes.
    ast_fn = cst.Module(
        body=[
            cst.FunctionDef(
                name=cst.Name(value=graph.name),
                params=cst.Parameters(params=args),
                body=cst.IndentedBlock(body=stmts + returns),
            )
        ]
    )

    code = ast_fn.code
    print(code)
    exec(code, namespace)
    fn = namespace[graph.name]

    return fn


def compile_op(node: Op, graph: GraphicalModel):
    """Compile an Op by recursively compiling and including its
    upstream nodes.
    """
    op_args = []
    op_kwargs = {}
    for predecessor in graph.predecessors(node):

        # I am not sure what this is about, consider deleting.
        if graph[predecessor][node] == {}:
            pass

        # If a predecessor has a name, it is either a random or
        # a deterministic variable. We only need to reference its
        # name here.
        if predecessor.name is not None:
            pred_ast = cst.Name(value=predecessor.name)
        else:
            pred_ast = compile_op(predecessor, graph)

        # To rebuild the node's CST we need to pass the compiled
        # CST of the arguments as arguments to the generator function.
        #
        # !! Important !!
        #
        # The compiler feeds the arguments in the order in which they
        # were added to the graph, not the order in which they were
        # given to `graph.add`. This is a potential source of mysterious
        # bugs and should be corrected.
        # This also means we do not feed repeated arguments several times
        # when we should.
        if graph[predecessor][node]["type"] == "arg":
            op_args.append(pred_ast)
        else:
            for key in graph[predecessor][node]["key"]:
                op_kwargs[key] = pred_ast

    print(node, op_args, op_kwargs)
    return node.to_ast(*op_args, **op_kwargs)
