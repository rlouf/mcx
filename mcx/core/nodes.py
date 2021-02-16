from dataclasses import dataclass
from typing import Callable, Optional

from .graph import GraphicalModel


@dataclass(frozen=True)
class Name:
    """Name of an attribute or a function."""

    cst_generator: Callable
    name: str


@dataclass(frozen=True)
class Placeholder:
    """Placeholder node.

    A Placeholder is a named node whose shape and value is not known until
    execution. Placeholders are collected at compilation time to be used as
    argument to the function being compiled.
    """

    cst_generator: Callable
    name: str
    is_random_variable: bool = False
    has_default: bool = False


class Constant(object):
    """Constant node.

    Constant nodes hold a value that is no modified during the
    program's execution, whether when computing the log-probability
    or sampling from prior & posterior distributions.
    """

    def __init__(self, cst_generator: Callable, name: Optional[str] = None):
        self.cst_generator = cst_generator
        self.name = name


class Op(object):
    """An Op node.

    Ops are operations that take one or many Ops or Vars as an input and
    return a new Var. In the following expression:

        >>> a = np.dot(X, np.ones(1))

    `np.dot` is a named Op (returns the variable 'a') that takes the Variable X
    and the Op `np.ones` as inputs.

    The parser returns an Op named `a` with the following ast stub:

        >>> ast.Call(
        ...     func=ast.Attribute(
        ...         value=ast.Name(id='np', ctx=ast.Load()),
        ...         attr='dot',
        ...         ctx=ast.Load()
        ...     ),
        ...     args=[],
        ...     keywords=[].
        ... )

    """

    def __init__(
        self,
        cst_generator,
        scope: Optional[str],
        name: Optional[str] = None,
        is_returned=False,
        do_sample: bool = False,
    ) -> None:
        self.name = name
        self.scope = scope
        self.cst_generator = cst_generator
        self.do_sample = do_sample
        self.is_returned = is_returned


class SampleOp(Op):
    """A SampleOp node.

    SampleOps represent random variables in the model. Unlike Ops, SampleOps
    are necessarily named.
    """

    def __init__(
        self,
        cst_generator: Callable,
        scope: str,
        name: str,
        distribution,
        is_returned=False,
    ) -> None:
        self.name = name
        self.scope = scope
        self.cst_generator = cst_generator
        self.distribution = distribution
        self.is_returned = is_returned


class SampleModelOp(SampleOp):
    """A SampleModelOp node.

    SampleOps represent random variables that are defined by a MCX model.
    """

    def __init__(
        self,
        cst_generator: Callable,
        scope: str,
        name: str,
        model_name: str,
        graph: GraphicalModel,
        is_returned=False,
    ) -> None:
        self.cst_generator = cst_generator
        self.scope = scope
        self.name = name
        self.model_name = model_name
        self.graph = graph
        self.is_returned = is_returned


@dataclass(frozen=True)
class FunctionOp:
    """Function node.

    Stores a standard python function as is.
    """

    cst_generator: Callable
    name: str


@dataclass(frozen=True)
class ModelOp(object):
    """Model node.

    Stores a MCX model (a function decorated with @mcx.model) as is.
    """

    cst_generator: Callable
    name: str
