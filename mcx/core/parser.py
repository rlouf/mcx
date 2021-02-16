"""Parse the model definition into a MCX graph.

When the user defines a model in MCX she uses the symbol `<~` to denote random
variable assignment, and the decorator `@mcx.model` to denote the definition of
a multivariate distribution. The constructs do not exist in Python and we parse
them into SampleOps and GraphicalModels respectively. Other python constructs
are transformed into Ops using a surjective mapping.

The ensuing graphical model can then be manipulated at runtime, and used to compile
samplers and logpdfs.

"""
from functools import partial
import inspect
import textwrap
from typing import Callable, Dict, List, Optional, Union

import libcst as cst
import networkx as nx

import mcx
from mcx.core.graph import GraphicalModel
from mcx.core.nodes import Constant, Op, Placeholder, SampleOp, Name

MODEL_BADLY_FORMED_ERROR = SyntaxError(
    "A MCX model should be defined in a single function. This exception should not be raised."
    " Please file an issue on https://github.com/rlouf/mcx."
)

MULTIPLE_RETURNED_VALUES_ERROR = SyntaxError(
    "Only one variable allowed on the left-hand-side of a random variable assignment "
    " , several were provided."
)

NO_SAMPLE_DISTRIBUTION_ERROR = SyntaxError(
    "Only instances of Distribution object (which includes distributions and models)"
    " are allowed on the right-hand-side of a random variable assignment."
)

ONLY_RETURN_NAMED_VARS_ERROR = SyntaxError(
    "Models can only return named variable."
    " Expressions such as `return a * b` are forbidden."
    " Do `name = a * b` then `return name` instead."
)

MODEL_OUTPUT_MATCH_ERROR = ""


def parse(model: Callable, namespace: Dict) -> GraphicalModel:
    """Build a graphical representation of the model definition."""
    source = inspect.getsource(model)
    source = textwrap.dedent(source)
    tree = cst.parse_module(source)

    definition_visitor = ModelDefinitionParser(namespace)
    tree.visit(definition_visitor)
    graph = definition_visitor.graph

    return graph


class ModelDefinitionParser(cst.CSTVisitor):
    def __init__(self, namespace):
        self.scope = None
        self.namespace = namespace

        self.sample_deterministic = True
        self.named_variables = {}

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """There's an issue if a model is defined inside the model.

        The main model definition
        ~~~~~~~~~~~~~~~~~~~~~~~~~

            >>> @mcx.model
            ... def my_model(*args):
            ...     # do things
            ...     return

        A regular function
        ~~~~~~~~~~~~~~~~~~~

            >>> @mcx.model
            ... def my_model(*args):
            ...     x <~ Normal(0, 1)
            ...     y <~ Normal(0, 1)
            ...
            ...     def add(a, b):
            ...         return a + b
            ...
            ...     z = add(x, y)
            ...     return z

        A closure
        ~~~~~~~~~

        It is perfectly reasonable (and is necessary to work with nonparametrics) to define
        a model like the following:

            >>> @mcx.model
            ... def mint_coin():
            ...     p <~ Beta(1, 1)
            ...
            ...     @mcx.model
            ...     def coin():
            ...         head <~ Bernoulli(p)
            ...         return head
            ...
            ...     return coin

        A submodel
        ~~~~~~~~~~

            >>> @mcx.model
            ... def linreg(x):
            ...     scale <~ HalfCauchy(0, 1)
            ...
            ...     @mcx.model
            ...     def Horseshoe(mu=0, tau=1., s=1.):
            ...         scale <~ HalfCauchy(0, s)
            ...         noise <~ Normal(0, tau)
            ...         res = scale * noise + mu
            ...         return res
            ...
            ...     coefs <~ Horseshoe(np.zeros(x.shape[1]))
            ...     predictions <~ Normal(np.matmul(x, coefs), scale)
            ...     return predictions

        """

        # Leave standard python functions alone.
        #
        # Actually, they will need to be included if they are called within the
        # model. This is an annoying corner case that can probably be handled by
        # a simple FunctionOp.

        # if not is_model_definition(node, self.namespace):
        # return

        # Each time we enter a model definition we create a new Graphical Model
        # which is returned after the definition's children have been visited.
        # The current version does not support nested graph but will.
        self.graph = GraphicalModel()
        self.graph.name = node.name.value
        self.scope = node.name.value

        def to_ast(name, default):
            return cst.Param(
                name=cst.Name(value=name),
                default=default,
            )

        function_args = node.params.params
        for i, argument in enumerate(function_args):
            name = argument.name.value
            default = argument.default
            node = Placeholder(name, partial(to_ast, name, default))
            self.named_variables[name] = node
            self.graph.add(node)

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        """Checks for comments after deterministic assignments."""
        comment = node.trailing_whitespace.comment
        if comment is not None and "sample: ignore" in comment.value:
            self.sample_deterministic = False

    def leave_SimpleStatement(self, node: cst.SimpleStatementLine) -> None:
        self.sample_deterministic = True

    # ----------------------------------------------------------------
    #          PARSE DETERMINISTIC ASSIGNMENTS (Named Ops)
    # ----------------------------------------------------------------

    def visit_Assign(self, node: cst.Assign) -> None:
        """A new named Op or Constant.

        There are three broad situations to take into consideration. First,
        numerical constants:

            >>> a = 0

        which are parsed into a Constant node. Then variable transformations:

            >>> a = np.dot(x, y)
            ... a = w + 2
            ... a = x < 0
            ... a = ~x

        Which can appear as function calls or python operations `ast.Add`,
        `ast.Mul`, etc. They are parsed into Op nodes.

        Finally, there are constants that are not encoded by a `ast.Constant`
        nodes. Numpy arrays for instance:

            >>> a = np.ones(10)

        These can be distinguished from regular Ops by the fact that they are
        not linked to a named variable. They are parsed in a Constant node as
        well.
        """
        op = self.recursive_visit(node.value)
        op.name = node.targets[0].target.value  # We assume all Ops have single outputs
        self.named_variables[op.name] = op

    def recursive_visit(self, node) -> Union[Constant, Op]:
        """Recursively visit the expression on the right-hand side of the assignment
        statement and populate the graph with the corresponding nodes.

        There are two cases we need to handle in a cleaner way:
        - Slices. There is no reason to detail the succession of nodes in the graph per se.
        - Functions like `np.dot` where `np` and `dot` are stored in separate nodes. This is stupid.

        Note
        ----
        This function could be re-rewritten with functools'
        `singledispatchmethod` but it is not available for Python 3.7. While
        there is a library that does backward-compatibility I prefered to avoid adding
        a dependency.

        """

        if isinstance(node, cst.Name):
            """If the node is a name it corresponds to a named Op, a constant
            or a Placeholder. In which case it should be registered in
            `named_variables`.

            Otherwise, this probably corresponds to an attribute node and this
            case will eventually disappear when this refactor is done.
            """
            try:
                name = self.named_variables[node.value]
            except KeyError:
                name = Name(node.value, lambda: node)
            return name

        if isinstance(node, cst.BaseNumber):
            new_node = Constant(lambda: node)
            return new_node

        if isinstance(node, cst.Attribute):
            value = self.recursive_visit(node.value)
            attr = self.recursive_visit(node.attr)

            def to_ast(value, attr):
                return cst.Attribute(value=value, attr=attr)

            op = Op(to_ast)
            self.graph.add(op, value, attr)
            return op

        if isinstance(node, cst.Arg):
            value = self.recursive_visit(node.value)

            def to_ast(value):
                return cst.Arg(
                    value=value,
                    keyword=node.keyword,
                )

            op = Op(to_ast)
            self.graph.add(op, value)
            return op

        if isinstance(node, cst.Call):
            args = [self.recursive_visit(arg) for arg in node.args]
            func = self.recursive_visit(node.func)

            def to_ast(*args, **kwargs):
                """This will need to be adapted.
                Will break when we use of keyword arguments."""
                return cst.Call(
                    func=kwargs["__func__"],
                    args=args,
                )

            op = Op(to_ast)
            self.graph.add(op, *args, __func__=func)

            return op

        if isinstance(node, cst.Subscript):
            value = self.recursive_visit(node.value)
            slice_elements = [self.recursive_visit(s) for s in node.slice]

            def to_ast(value, *slice_elements):
                return cst.Subscript(
                    value=value,
                    slice=slice_elements,
                )

            op = Op(to_ast)
            self.graph.add(op, value, *slice_elements)

            return op

        if isinstance(node, cst.SubscriptElement):
            sl = self.recursive_visit(node.slice)

            def to_ast(sl):
                return cst.SubscriptElement(slice=sl)

            op = Op(to_ast)
            self.graph.add(op, sl)

            return op

        if isinstance(node, cst.Index):
            value = self.recursive_visit(node.value)

            def to_ast(value):
                return cst.Index(value=value)

            op = Op(to_ast)
            self.graph.add(op, value)

            return op

        if isinstance(node, cst.BinaryOperation):
            left = self.recursive_visit(node.left)
            right = self.recursive_visit(node.right)

            def to_ast(left, right):
                return cst.BinaryOperation(
                    left=left,
                    operator=node.operator,
                    right=right,
                )

            op = Op(to_ast)
            self.graph.add(op, left, right)
            return op

        if isinstance(node, cst.UnaryOperation):
            expression = self.recursive_visit(node.expression)

            def to_ast(expression):
                return cst.UnaryOperation(
                    operator=node.operator,
                    expression=expression,
                )

            op = Op(to_ast)
            self.graph.add(op, expression)
            return op

        raise TypeError(
            f"The CST node {node.__class__.__name__} is currently not supported in MCX. "
            "Please open an issue on https://github.com/rlouf/mcx so we can integrate it."
        )

    # ----------------------------------------------------------------
    #         PARSE RANDOM VARIABLE ASSIGNMENTS (Sample Ops)
    # ----------------------------------------------------------------

    def visit_Comparison(self, node: cst.Comparison) -> None:
        """Parse random variables.

        Random variable assignments are represented in MCX by the symbol `<~`
        which in Python syntax is the combination of the `Lt` and `Invert`
        operators.
        We thus replace the succession of `<` and `~` in the Python AST by a
        single `Sample` node in MCX's AST. We also replace the `Name` node
        that designates the random variable by a `RVName` node.

        The object on the right-hand side of the operator is a mcx model:

            >>> coef <~ Horseshoe(1.)

        Otherwise we assume that the the operator is a distribution:

            >>> a <~ Normal(0, 1)
            ... a <~ dist.Normal(0, 1)

        Then we build a SampleOp from the expression.


        Then we merge the model's implementation with that of the current model.
        """
        if len(node.comparisons) != 1:
            return

        operator = node.comparisons[0].operator
        comparator = node.comparisons[0].comparator
        if isinstance(operator, cst.LessThan) and isinstance(
            comparator, cst.UnaryOperation
        ):
            if isinstance(node.left, cst.Tuple):
                raise MULTIPLE_RETURNED_VALUES_ERROR

            variable_name = node.left.value

            fn_call_path = unroll_call_path(comparator.expression.func)
            fn_obj = eval(fn_call_path, self.namespace)
            if isinstance(fn_obj, mcx.model):
                sample_op = self.graph.merge(
                    variable_name, fn_obj.graph
                )  # returns the return Op
            else:
                op = self.recursive_visit(comparator.expression)
                sample_op = SampleOp(variable_name, op.to_ast)
                mapping = {op: sample_op}
                self.graph = nx.relabel_nodes(self.graph, mapping)

            self.named_variables[variable_name] = sample_op

    # ----------------------------------------------------------------
    #                   TAG RETURNED VARIABLES
    # ----------------------------------------------------------------

    def visit_Return(self, node: cst.Return) -> None:
        """We need to tag variables as returned leaves in the graph.
        Return is an op and should be the only on in the graph.
        """

        if not isinstance(node.value, cst.Name):
            raise ONLY_RETURN_NAMED_VARS_ERROR

        returned_name = self.named_variables[node.value.value]

        def to_ast():
            return cst.Name(value=returned_name)

    # ----------------------------------------------------------------
    #           EXPLICITLY EXCLUDE CONTROL FLOW (for now)
    # ----------------------------------------------------------------

    def visit_If(self, node: cst.If):
        raise SyntaxError(
            "Probabilistic programs with stochastic support cannot be defined "
            "using python's control flow yet. Please use jax.lax.cond or jax.lax.switch instead. "
        )

    def visit_For(self, node: cst.For):
        raise SyntaxError(
            "Probabilistic programs cannot use python's control flow constructs yet "
            "Please use jax.lax.scan (preferably), jax.lax.fori_loop or jax.lax.while instead. "
        )


def is_model_definition(node: cst.FunctionDef, namespace: Dict) -> bool:
    """Determines whether a function is a model definition.

    A function is a model definition if it is decorated by the mcx.model class.
    This is not completely straightforward as the class can be called in many
    different ways, per python's import rules:

        >>> import mcx
        ... @mcx.model

        >>> import mcx as pymc3
        ... @pymc3.model

        >>> from mcx import model
        ... @model

        >>> import mcx.model as turing_machine
        ... @turing_machine

    we handle all these situations using python's introspection capabilities and
    the namespace we captured when the model was called.

    """
    for decorator in node.decorators:

        if isinstance(decorator, cst.Name):
            obj = namespace[decorator.id]
            if obj.__module__ == "mcx" and obj.__name__ == "model":
                return True
        elif isinstance(decorator, cst.Attribute):
            mod = namespace[decorator.value.id]
            obj = getattr(mod, decorator.attr)
            if mod.__name__ == "mcx" and obj.__name__ == "model":
                return True

    return False


def unroll_call_path(
    node: Union[cst.Attribute, cst.Call], name: Optional[List[str]] = None
) -> str:
    if not name:
        name = []

    if isinstance(node, cst.Name):
        name.insert(0, node.value)
        return ".".join(name)
    elif isinstance(node, cst.Attribute):
        name.insert(0, node.attr.value)
        new_node = node.value
        return unroll_call_path(new_node, name)
    else:
        raise ValueError(
            "Error while getting the right-hand-side object's name: expected `ast.Name`"
            f"or `ast.Attribute`, got {node}"
        )


def name_to_arg(node: cst.Name) -> cst.Arg:
    """Convert a Name node to an Arg mode."""
    pass
