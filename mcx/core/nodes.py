from typing import Callable, Optional


class Constant(object):
    """Constant node.

    Constant nodes hold a value that is not modified during the program's
    execution. Will eventually hold expressions such as `np.ones(10)` that
    do not contain any name.
    """

    def __init__(self, ast_generator: Callable, name: str = None) -> None:
        self.name = name
        self.to_ast = ast_generator


class Name(object):
    """This is temporarily here to handle the attributes. Will disappear 
    eventually.
    """

    def __init__(self, name: str, ast_generator: Callable) -> None:
        self.name = name
        self.to_ast = ast_generator


class Placeholder(object):
    """Placeholder node.

    A Placeholder node is a named node whose shape and value is unknown util
    execution. Placeholders are collected during compilation to be added as
    arguments to the function.
    """

    def __init__(self, name: str, ast_generator: Callable, rv=False) -> None:
        self.name = name
        self.to_ast = ast_generator
        self.rv = rv


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
        ...         attr='dot', ctx=ast.Load()
        ...     ),
        ...     args=[],
        ...     keywords=[].
        ... )

    """

    def __init__(
        self, ast_generator, scope: Optional[str], name: Optional[str] = None, do_sample: bool = False
    ) -> None:
        self.name = name
        self.scope = scope
        self.to_ast = ast_generator
        self.do_sample = False


class SampleOp(Op):
    """A SampleOp node.

    SampleOps represent random variables in the model. Unlike Ops, SampleOps
    are necessarily named.
    """

    def __init__(self, name: str, scope: str, ast_generator: Callable) -> None:
        self.name = name
        self.scope = scope
        self.to_ast = ast_generator
