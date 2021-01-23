import libcst as cst
import networkx as nx

from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Constant, Op, Placeholder


def compile_graph(graph: GraphicalModel, namespace: dict, fn_name):
    """Compile MCX's graph into a python (executable) function."""

    # Model arguments.
    #
    # Arguments are passed in the following order:
    # 1. rng_key (if relevant);
    # 2. Random variables _in the order in which they appear in the model_
    # 3. The model's arguments and keyword arguments.
    maybe_rng_key = [
        compile_placeholder(node, graph)
        for node in graph.nodes
        if isinstance(node, Placeholder) and node.name == "rng_key"
    ]
    maybe_random_variables = [
        compile_placeholder(node, graph)
        for node in reversed(list(graph.nodes))
        if isinstance(node, Placeholder) and node.rv
    ]
    model_args = [
        compile_placeholder(node, graph)
        for node in graph.nodes
        if isinstance(node, Placeholder) and not node.rv and node.name != "rng_key"
    ]
    args = maybe_rng_key + maybe_random_variables + model_args

    # Every statement in the function corresponds to either a constant definition or
    # a variable assignment. We use a topological sort to respect the
    # dependency order.
    stmts = []
    returns = []
    for node in nx.topological_sort(graph):

        if node.name is None:
            continue

        if isinstance(node, Constant):
            stmt = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(value=node.name))],
                        value=node.to_ast(),
                    )
                ]
            )
            stmts.append(stmt)

        if isinstance(node, Op):
            stmt = cst.SimpleStatementLine(
                body=[
                    cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(value=node.name))],
                        value=compile_op(node, graph),
                    )
                ]
            )
            stmts.append(stmt)

            if node.is_returned:
                returns.append(
                    cst.SimpleStatementLine(
                        body=[cst.Return(value=cst.Name(value=node.name))]
                    )
                )

    # Assemble the function's CST using the previously translated nodes.
    ast_fn = cst.Module(
        body=[
            cst.FunctionDef(
                name=cst.Name(value=fn_name),
                params=cst.Parameters(params=args),
                body=cst.IndentedBlock(body=stmts + returns),
            )
        ]
    )

    code = ast_fn.code
    exec(code, namespace)
    fn = namespace[fn_name]

    return fn, code


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

    return node.to_ast(*op_args, **op_kwargs)


def compile_placeholder(node: Placeholder, graph: GraphicalModel):
    """Compile a placeholder by fetching its default value."""
    default = []
    for predecessor in graph.predecessors(node):
        default.append(predecessor.to_ast())

    return node.to_ast(*default)
