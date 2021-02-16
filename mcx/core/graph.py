"""The McxST symbolic graph."""
import networkx as nx

from mcx.core.nodes import SampleOp, Op


class GraphicalModel(nx.DiGraph):
    """Intermediate representation of a probabilistic model in MCX."""

    def __init__(self):
        super().__init__()

    def add(self, node, *args, **kwargs):
        """Add a new node to the graph.

        When adding a new Op to the graph we also pass its input nodes so
        we can connect them in the graph.

        """
        if node in self.nodes:
            return node

        self.add_node(node)

        # if the node is an Op we also need to add its arguments
        if isinstance(node, Op):
            for i, arg in enumerate(args):
                arg_node = self.add(arg)  # add to graph if does not exist
                if self.has_edge(arg_node, node):
                    self[arg_node][node]["name"]["position"].append(i)
                else:
                    self.add_edge(arg_node, node, type="arg", position=[i])

            for key, arg in kwargs.items():
                target = self.add(arg)  # add to graph if does not exist
                if self.has_edge(target, node):
                    self[target][node]["key"].append(key)
                else:
                    self.add_edge(target, node, type="kwarg", key=[key])

        return node

    def merge(self, name: str, model: "GraphicalModel"):
        """Merge a model with the current one.
        """
        pass

    def do(self, **kwargs):
        """Apply the do-operator to the graph:

        1. Simplify the graph by removing parent nodes. Send warning when one
           of the variables disappear because of another one.
        2. Add partial application logic for when the function is compiled.
        """
        pass

    @property
    def random_variables(self):
        """Returns the random variable nodes, represented by the "Sample" Op.
        """
        return [node for node in nx.topological_sort(self) if type(node) == SampleOp]
