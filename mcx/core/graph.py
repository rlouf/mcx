"""The McxST symbolic graph."""
from typing import Union

import networkx as nx

from mcx.core.nodes import Constant, SampleOp, Op, Placeholder


class GraphicalModel(nx.DiGraph):
    """Intermediate representation of a probabilistic model in MCX."""

    def __init__(self):
        super().__init__()

    def add(self, node, *args, **kwargs) -> None:
        """Add a new node to the graph.

        When a new node is added we also draw edges from the nodes that are
        passed as arguments. These nodes are added in the graph if not already
        present.

        Parameters
        ----------
        node
            The Op or Variable to add to the graph.
        *args
            The Ops or Variables passed to the node as arguments.
        *kwargs
            The Ops or Variables passed to the node as keyword arguments

        """
        if node in self.nodes:
            return

        self.add_node(node)

        if isinstance(node, Placeholder):
            for i, arg in enumerate(args):
                self.add(arg)
                self.add_edge(arg, node, type="arg", position=[0])

        if isinstance(node, Op):
            for i, arg in enumerate(args):
                self.add(arg)
                if self.has_edge(arg, node):
                    self[arg][node]["name"]["position"].append(i)
                else:
                    self.add_edge(arg, node, type="arg", position=[i])

            for key, arg in kwargs.items():
                self.add(arg)
                if self.has_edge(arg, node):
                    self[arg][node]["key"].append(key)
                else:
                    self.add_edge(arg, node, type="kwarg", key=[key])

    def merge(self, assigned_name: str, posargs, kwargs, model: "GraphicalModel") -> Union["GraphicalModel", Placeholder]:
        """Merge a model with the current one.

        Parameters
        ----------
        assigned_name
            The name of the variable returned by the model being merged
            in the current model.
        model
            Graph of the model that we want to merge.
        """

        # Find the model's leaf and rename it
        return_node = model.leaf
        return_node.name = assigned_name
        return_node.is_returned = False

        # We now need to translate the model's placeholders into constants.
        # We first check whether its value is provided when calling from
        # the current model, or whether it has a default value. Otherwise
        # raise the same error you would when not providing a positional
        # agument to a function.

        mapping = {}

        # Replace positional arguments by constants
        num_positional = len(posargs)
        placeholders = model.placeholders
        for arg, placeholder in zip(posargs, placeholders[:num_positional]):
            mapping[placeholder] = arg

        # Remaining placeholders
        remaining_placeholder = placeholders[num_positional:]
        missing_arguments = []
        for placeholder in remaining_placeholder:

            if model.in_degree(placeholder) == 0:  # not a keyword argument
                if placeholder.name in kwargs:
                    arg = kwargs[placeholder.name]
                    mapping[placeholder] = arg
                else:
                    missing_arguments.append(placeholder.name)
            else:  # a keyword argument
                if placeholder.name in kwargs:
                    arg = kwargs[placeholder.name]
                    mapping[placeholder] = arg
                else:
                    default = list(model.predecessors(placeholder))[0]
                    mapping[placeholder] = Constant(lambda: default.to_ast(), placeholder.name)

        if len(missing_arguments) > 0:
            num_missing = len(missing_arguments)
            maybe_s = 's' if num_missing > 1 else ''
            missing_names = ', '.join(missing_arguments)
            raise TypeError(f"{model.name}() missing {num_missing} argument{maybe_s}: {missing_names}")

        model = nx.relabel_nodes(model, mapping)
        merged_graph = nx.compose(model, self)

        return merged_graph, return_node

    def do(self, **kwargs):
        """Apply the do-operator to the graph:

        1. Simplify the graph by removing parent nodes. Send warning when one
           of the variables disappear because of another one.
        2. Add partial application logic for when the function is compiled.
        """
        pass

    @property
    def leaf(self):
        leaves = [node for node in self.nodes() if self.out_degree(node) == 0]
        if len(leaves) > 1:
            raise SyntaxError
        return leaves[0]

    def has_default_value(self, placeholder: Placeholder):
        return self.in_degree(placeholder) == 0

    @property
    def placeholders(self):
        return [node for node in self.nodes() if isinstance(node, Placeholder)]

    @property
    def random_variables(self):
        """Returns the random variable nodes, represented by the "Sample" Op.
        """
        return [node for node in self.nodes() if isinstance(node, SampleOp)]
