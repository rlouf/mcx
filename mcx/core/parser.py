"""Parse the model definition into a MCX graph.

When the user defines a model in MCX she uses the symbol `<~` to denote random
variable assignment, and the decorator `@mcx.model` to denote the definition of
a multivariate distribution. The constructs do not exist in Python and we parse
them into SampleOps and GraphicalModels respectively. Other python constructs
are transformed into Ops using a surjective mapping.

The ensuing graphical model can then be manipulated at runtime, and used to compile
samplers and logpdfs.

"""
import inspect
import textwrap
from collections import defaultdict
from functools import partial
from types import FunctionType
from typing import Dict, List, Optional, Tuple, Union

import libcst as cst
import networkx as nx

import mcx
from mcx.core.graph import GraphicalModel
from mcx.core.nodes import (
    Constant,
    FunctionOp,
    ModelOp,
    Name,
    Op,
    Placeholder,
    SampleModelOp,
    SampleOp,
)

MODEL_BADLY_FORMED_ERROR = (
    "a MCX model should be defined in a single function. This exception is completely unexpected."
    " Please file an issue on https://github.com/rlouf/mcx"
)

# TODO: Allow random variable assignments from models that return multiple variables
MULTIPLE_RETURNED_VALUES_ERROR = (
    "only one variable is allowed on the left-hand-side of a random variable assignment "
    " , several were provided"
)

TRANSFORMED_RETURNED_VALUE_ERROR = (
    "only random variables can be returned from the model, found a transformed variable instead. "
    "If you are interested in the posterior value of such a transformed variable, first sample "
    " from the posterior distribution of random variables and apply the transformation to the variables "
    " in the trace.\n"
    "If you are looking to condition on a transformed variable, however, this is not yet possible. Please "
    "open an issue on https://github.com/rlouf/mcx to signal your interest in having this feature"
)

DUPLICATE_VARIABLE_NAME_ERROR = "you cannot reuse the name of random variables."


def parse(model_fn: FunctionType) -> Tuple[GraphicalModel, dict]:
    """Parse the model definition to build a graphical model.

    Parameter
    ---------
    model
        A live function object that contains the model definition.

    Returns
    -------
    The intermediate representation of the model.

    """
    source = inspect.getsource(model_fn)
    source = textwrap.dedent(source)  # TODO: do we need this with libcst?
    tree = cst.parse_module(source)

    namespace = model_fn.__globals__

    definition_visitor = ModelDefinitionParser(namespace)
    tree.visit(definition_visitor)
    graph = definition_visitor.graph

    return graph, namespace


class ModelDefinitionParser(cst.CSTVisitor):
    """Parses the model definition's Concrete Syntax Tree.

    MCX models are expressed in the form of a function with the `@mcx.model`
    decorator. MCX then parses the definition's code into an intermediate
    representation, which is a graph where sample and deterministic operations
    as well as the function's arguments are named nodes while intermediate
    operations are part of the graph but unnamed.

    This approach is similar to that of Theano. But unlike Theano, we do not
    build the graph at runtime using side-effects. Instead, when the model is
    instantiated its code source is read and translated into a graph. Every
    subsequent operation is an operation on this graph.

    This class contains all the parsing and graph building logic. Whenever a
    sample statement is encountered, we perform a recursive visit of the
    variables used to instantiate the distribution.  The recursion stops
    whenever we encounter a constant or a named variable.  We then add node in
    the reverse order to the graph; nodes contain a function that can
    reconstruct the corresponding CST node when called with arguments.

    Say the parser encounters the following line:

        >>> var <~ Normal(0, x)

    where `x` is an argument to the model. The recursion stops immediately
    since both arguments are either a constant or a named node. So we create a
    new `SampleOp` with name `var` and a cst_generator funtion defined as:

        >>> def cst_generator(*args):
        ...      return cst.Call(
        ...          func='Normal',
        ...          args=args
        ...      )

    And we create an edge between the constant `0` and this SampleOp, the
    placeholder `x` and this SampleOp. Each edge is indexed by the position of
    the argument. All the compiler has to do is to traverse the graph in
    topological order, translate `0` and `x` to their equivalent CST nodes, and
    pass these nodes to var's `cst_generator` function when translating it into its
    AST equivalent.

    When parsing the following:

        >>> var <~ Normal(0, f(x))

    The procedure is similar except we first add the unnamed `f` node to the
    graph and the edge from `x` to `f` before adding the `var` node and the
    edges 0 -> `var` and `f` -> `var`.

    There are a few potential pitfalls that we need to be aware of:

    1. When a random variable is distributed following another MCX model, the
    model is merged with the current one. Since namespaces can overlap we add a
    scope for variables, named after the model which introduced the variables.
    Furthermore, since the same model can be used several times (this would be
    common in deep learning), we append the model's current number of occurences
    to the scope name.

    2. As discussed in the docstring of `visit_FunctionDef` below, functions are
    to be treated with care as they can either be standard functions, models
    that are called within the current model, or models that are defined via
    a closure.

    Attributes
    ----------
    current_scope:
        Name of the model being currently parsed.
    scopes:
        Counts the number of times each scope has been encountered. Important in
        situations like deep learning where the same module can be called
        several times.
    namespace:
        Dictionnary that contains the global namespace in which the model is
        called.
    sample_this_op:
        Whether the Op currently being traversed should appear in forward sampling
        and whether we should include them in the posterior samples.
    named_variables:
        Dictionary that associates the name of each Op to its node.

    """

    def __init__(self, namespace: Dict):
        self.namespace = namespace
        self.current_scope: Optional[str] = None
        self.scopes: Dict = defaultdict(int)
        self.named_variables: Dict = {}
        self.sample_this_op = True

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Visit a function definition.

        When we traverse the Concrete Syntax Tree of a MCX model, a function definition
        can represent several objects.

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

        We can even have nested submodels.

        """

        # Standard python functions defined within a model need to be included
        # as is in the resulting source code. So do submodels.
        if hasattr(self, "graph"):
            if is_model_definition(node, self.namespace):
                graph_node = ModelOp(lambda: node, node.name)
                name = node.name
            else:
                graph_node = FunctionOp(lambda: node, node.name)
                name = node.name

            self.graph.add(graph_node)
            self.named_variables[node.name] = graph_node
            return False  # don't visit the node's children

        # Each time we enter a model definition we create a new GraphicalModel
        # which is returned after the definition's children have been visited.
        # The current version does not support nested models but will.
        self.graph = GraphicalModel()
        self.graph.namespace = self.namespace
        self.graph.name = node.name.value
        self.scope = node.name.value

        def argument_cst(name, default=None):
            return cst.Param(cst.Name(name), default=default)

        function_args = node.params.params
        for _, argument in enumerate(function_args):
            name = argument.name.value

            try:  # parse argument default value is any
                default = self.recursive_visit(argument.default)
                node = Placeholder(partial(argument_cst, name), name, False, True)
                self.graph.add(node, default)
            except TypeError:
                node = Placeholder(partial(argument_cst, name), name)
                self.graph.add(node)

            self.named_variables[name] = node

    # ----------------------------------------------------------------
    #                        PARSE COMMENTS
    # ----------------------------------------------------------------

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        """Read comments.

        LibCST includes each assignment statement in a statement line, to which
        comments are attached.

        We include an experimental feature which consists in allowing the user
        to ignore some deterministic variables in the trace by adding a
        `# sample: ignore` comment on the line it is defined.

        """
        comment = node.trailing_whitespace.comment
        if comment is not None and "sample: ignore" in comment.value:
            self.sample_this_op = False

    def leave_SimpleStatementLine(self, _) -> None:
        """Re-enable the sampling of deterministic variables after parsing the current line."""
        self.sample_this_op = True

    # ----------------------------------------------------------------
    #          PARSE DETERMINISTIC ASSIGNMENTS (named Ops)
    # ----------------------------------------------------------------

    def visit_Assign(self, node: cst.Assign) -> None:
        """Visit named Ops and Constants.

        This method parses assignment expressions such as:

            >>> x = f(a)

        We parse the right-hand side recursively to create the Op and its
        parent which can be Constants, Placeholders or Ops.

        TODO: For the sake of simplicity we currently explicitly disallow
        expressions that return a tuple. It should be possible to define
        multioutput ops to handle this situation.

        There are three broad situations to take into consideration. First,
        numerical constants:

            >>> a = 0

        which are parsed into a Constant node. Then transformations that include
        function call, binary operatations, unary operations and result of a
        comparison.

            >>> a = np.dot(x, y)
            ... a = w + 2
            ... a = x < 0
            ... a = ~x

        These are parsed into `Op` nodes.

        Finally, there are constants that are not encoded by a `ast.Constant`
        nodes. Numpy arrays for instance:

            >>> a = np.ones(10)

        These can be distinguished from regular Ops by the fact that they are
        not linked to a named variable. It is however hard to guess that these
        expressions are constant when parsing the tree. We thus parse them into
        a named `Op` node, and rely from a further simplication step to turn
        them into a constant.

        """
        op = self.recursive_visit(node.value)

        # We restrict ourselves to single-output ops
        if len(node.targets) > 1:
            raise SyntaxError(MULTIPLE_RETURNED_VALUES_ERROR)

        op.name = node.targets[0].target.value
        self.named_variables[op.name] = op

    # ----------------------------------------------------------------
    #         PARSE RANDOM VARIABLE ASSIGNMENTS (Sample Ops)
    # ----------------------------------------------------------------

    def visit_Comparison(self, node: cst.Comparison) -> None:
        """Parse sample statements.

        Sample statements are represented in MCX by the symbol `<~` which is
        the combination of the `Lt` and `Invert` operators in Python. We thus
        translate the succession of `<` and `~` in the Python AST into a single
        `SampleOp` in MCX's graph.

        The parser can encounter two situations. First, the object on the
        right-hand side of the 'sample' operator is a MCX model:

            >>> coef <~ Horseshoe(1.)

        In this case the parser instantiates the model, parses it and merges
        its graph with the current graph.

        Otherwise the assumes that the object on the right hand-side of the operator has
        the same API as the `mcx.distributions.Distribution` class

            >>> a <~ Normal(0, 1)
            ... a <~ dist.Normal(0, 1)

        And builds a SampleOp from the expression.

        """
        if len(node.comparisons) != 1:
            return

        operator = node.comparisons[0].operator
        comparator = node.comparisons[0].comparator
        if isinstance(operator, cst.LessThan) and isinstance(
            comparator, cst.UnaryOperation
        ):
            if isinstance(node.left, cst.Tuple):
                raise SyntaxError(MULTIPLE_RETURNED_VALUES_ERROR)

            variable_name = node.left.value
            expression = comparator.expression

            if variable_name in self.named_variables:
                raise SyntaxError(DUPLICATE_VARIABLE_NAME_ERROR)

            # Eval the object on the righ-hand side of the <~ operator
            # This eval is necessary to check whether object sampled
            # from is a model or a distribution. And in the former case
            # to merge to the current graph.
            try:
                fn_call_path = unroll_call_path(expression.func)
            except AttributeError as err:
                raise SyntaxError(
                    "Expressions on the right-hand-side of <~ must be models or distributions. "
                    f"Found the node {expression} instead."
                ) from err

            fn_obj = eval(fn_call_path, self.namespace)
            if isinstance(fn_obj, mcx.model):
                op = self.recursive_visit(comparator.expression)
                sample_op = SampleModelOp(
                    op.cst_generator,
                    self.scope,
                    variable_name,
                    fn_call_path,
                    fn_obj.graph,
                )
                self.graph = nx.relabel_nodes(self.graph, {op: sample_op})
            elif issubclass(fn_obj, mcx.distributions.Distribution):
                op = self.recursive_visit(comparator.expression)
                sample_op = SampleOp(
                    op.cst_generator, self.scope, variable_name, fn_obj
                )
                self.graph = nx.relabel_nodes(self.graph, {op: sample_op})
            else:
                raise SyntaxError(
                    "Expressions on the right-hand-side of <~ must be models or distributions. "
                    f"Found {fn_call_path} instead."
                )

            self.named_variables[variable_name] = sample_op

    # ----------------------------------------------------------------
    #            RECURSIVELY PARSE THE RHS OF STATEMENTS
    # ----------------------------------------------------------------

    def recursive_visit(self, node) -> Union[Constant, Op]:
        """Recursively visit the node and populate the graph with the traversed nodes.

        The recursion ends when the CST node being visited is a `Name` or a
        `BaseNumber` node.  While we follow strictly libcst's CST
        decomposition, it may be desirable to simplify the graph for our
        purposes. For instance:

        - Slices and subscripts. There is no reason to detail the succession of
          nodes in the graph. It can either be a constant (only depends on numerical constants),
          or a function of other variables.
        - Functions like `np.dot`. `np` and `dot` are currently store in different Ops. We should
          merge these.

        TODO: Implement a function that takes a GraphicalModel and applies
        these simplifications. This will be necessary when sampling
        deterministic functions.

        Note
        ----
        This function could be re-rewritten using functools'
        `singledispatchmethod` but it is not available for Python 3.7. While
        there is a library that does backward-compatibility I prefered to avoid
        adding a dependency.

        """
        if isinstance(node, cst.Name):
            """If the node corresponds to a placeholder or a named op its name
            should be registered in `named_variables`. Otherwise it corresponds
            to the name of an attribute.

            """
            try:
                name = self.named_variables[node.value]
            except KeyError:
                name = Name(lambda: node, node.value)
            return name

        if isinstance(node, cst.BaseNumber):
            new_node = Constant(lambda: node)
            return new_node

        # Parse function calls

        if isinstance(node, cst.Call):
            func = self.recursive_visit(node.func)
            args = [self.recursive_visit(arg) for arg in node.args]

            def to_call_cst(*args, **kwargs):
                # I don't exactly remember why we pass the `func` as a keyword
                # argument, but I think it has something to do with the fact
                # that at compilation the arguments are passed in the order they
                # were introduced in the graph, and nodes are deleted/re-inserted
                # when transforming to get logpdf and samplers.
                func = kwargs["__name__"]
                return cst.Call(func, args)

            op = Op(to_call_cst, self.scope)
            self.graph.add(op, *args, __name__=func)
            return op

        if isinstance(node, cst.Arg):
            value = self.recursive_visit(node.value)

            def to_arg_cst(value):
                return cst.Arg(value, node.keyword)

            op = Op(to_arg_cst, self.scope)
            self.graph.add(op, value)
            return op

        if isinstance(node, cst.Attribute):
            value = self.recursive_visit(node.value)
            attr = self.recursive_visit(node.attr)

            def to_attribute_cst(value, attr):
                return cst.Attribute(value, attr)

            op = Op(to_attribute_cst, self.scope)
            self.graph.add(op, value, attr)
            return op

        # Parse slices and subscripts

        if isinstance(node, cst.Subscript):
            value = self.recursive_visit(node.value)
            slice_elements = [self.recursive_visit(s) for s in node.slice]

            def to_subscript_cst(value, *slice_elements):
                return cst.Subscript(value, slice_elements)

            op = Op(to_subscript_cst, self.scope)
            self.graph.add(op, value, *slice_elements)

            return op

        if isinstance(node, cst.SubscriptElement):
            sl = self.recursive_visit(node.slice)

            def to_subscript_element_cst(sl):
                return cst.SubscriptElement(sl)

            op = Op(to_subscript_element_cst, self.scope)
            self.graph.add(op, sl)

            return op

        if isinstance(node, cst.Index):
            value = self.recursive_visit(node.value)

            def to_index_cst(value):
                return cst.Index(value)

            op = Op(to_index_cst, self.scope)
            self.graph.add(op, value)

            return op

        # Parse Binary and Unary operations

        if isinstance(node, cst.BinaryOperation):
            left = self.recursive_visit(node.left)
            right = self.recursive_visit(node.right)

            def to_binary_operation_cst(left, right):
                return cst.BinaryOperation(left, node.operator, right=right)

            op = Op(to_binary_operation_cst, self.scope)
            self.graph.add(op, left, right)

            return op

        if isinstance(node, cst.UnaryOperation):
            expression = self.recursive_visit(node.expression)

            def to_unary_operation_cst(expression):
                return cst.UnaryOperation(node.operator, expression)

            op = Op(to_unary_operation_cst, self.scope)
            self.graph.add(op, expression)

            return op

        # In case we missed an important statement or expression leave a friendly error
        # message and redirect the user to the issue tracker to let us know.
        raise TypeError(
            f"The CST node {node.__class__.__name__} is currently not supported by MCX's parser. "
            "Please open an issue on https://github.com/rlouf/mcx so we can integrate it."
        )

    # ----------------------------------------------------------------
    #                   TAG RETURNED VARIABLES
    # ----------------------------------------------------------------

    def visit_Return(self, node: cst.Return) -> None:
        """Visit the return statement.

        We mark the referenced named Op as returned.
        """
        value = node.value
        if not isinstance(value, cst.Name):
            raise SyntaxError(MULTIPLE_RETURNED_VALUES_ERROR)

        returned_name = node.value.value

        try:
            returned_node = self.named_variables[returned_name]
            if not isinstance(returned_node, SampleOp):
                raise SyntaxError(TRANSFORMED_RETURNED_VALUE_ERROR)

            returned_node.is_returned = True
        except KeyError:
            raise NameError(f"name '{returned_name}' is not defined")
        except SyntaxError:
            raise

    # ----------------------------------------------------------------
    #           EXPLICITLY EXCLUDE CONTROL FLOW (for now)
    #
    # We forbid the use of Python's control flow primitives, due to
    # the incompatibility with JAX's tracing when the operands or number
    # of loops is determined at runtime. We could in a first time relax
    # this restriction for simple cases before proposing a Python-to-JAX
    # translation.
    # ----------------------------------------------------------------

    def visit_If(self, _):
        raise SyntaxError(
            "Probabilistic programs with stochastic support cannot be defined "
            "using python's control flow yet. Please use jax.lax.cond or jax.lax.switch instead. "
        )

    def visit_For(self, _):
        raise SyntaxError(
            "Probabilistic programs cannot use python's control flow constructs yet "
            "Please use jax.lax.scan (preferably), jax.lax.fori_loop or jax.lax.while instead. "
        )


def is_model_definition(node: cst.FunctionDef, namespace: Dict) -> bool:
    """Determines whether a function is a model definition.

    A function is a model definition if it is decorated by the mcx.model class.
    This is not completely straightforward to determine as the class can be
    called in many different ways, per python's import rules:

        >>> import mcx
        ... @mcx.model

        >>> import mcx as pymc3
        ... @pymc3.model

        >>> from mcx import model
        ... @model

        >>> import mcx.model as turing_machine
        ... @turing_machine

    We handle all these situations using python's introspection capabilities and
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
    """Unroll a function's call path.

    When we call a function using a path from the module in which it is defined
    the resulting CST is a succession of nested `Attribute` statement. This
    function unrolls these statement to get the full call path. For instance
    `np.dot`, `mcx.distribution.Normal`.

    Example
    -------

        >>> expr = cst.Call(cst.Attribute("mcx", cst.Attribute("distribution", "Normal")), None)
        >>> unroll_call_path(expr)
        "mcx.distribution.Normal"

    """
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
