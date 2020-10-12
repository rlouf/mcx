"""Static checks regarding validity of distribution, etc
should happen here since it will be the inteface when we
modify the graph at runtime.
"""
import ast
from typing import List, Optional, Union

import astor

from mcx.distributions import Distribution


class Argument(object):
    def __init__(self, name: str, default: Optional[ast.expr] = None):
        self.name = name
        self.default_value = default
        self.is_returned = False

    def to_logpdf(self):
        return ast.arg(arg=self.name, annotation=None)

    def to_logpdf_iadd(self):
        return ast.arg(arg=self.name, annotation=None)

    def to_sampler(self):
        return ast.arg(arg=self.name, annotation=None)


class RandVar(object):
    def __init__(
        self,
        name: str,
        distribution: Distribution,
        args: Optional[List[Union[int, float]]],
        is_returned: bool,
    ):
        self.name = name
        self.distribution = distribution
        self.args = args
        self.is_returned = is_returned

    def __str__(self):
        return "{} ~ {}".format(self.name, astor.code_gen.to_source(self.distribution))

    def to_logpdf(self):
        """Returns the AST corresponding to the expression

        logpdf_{name} = {distribution}.logpdf_sum(*{arg_names})
        """
        return ast.Assign(
            targets=[ast.Name(id=f"logpdf_{self.name}", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution,
                    attr="logpdf_sum",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=self.name, ctx=ast.Load())],
                keywords=[],
            ),
        )

    def to_logpdf_iadd(self):
        """Returns the AST corresponding to the expression

        logpdf += {distribution}.logpdf_sum(*{arg_names})
        """
        return ast.AugAssign(
            target=ast.Name(id="logpdf", ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution,
                    attr="logpdf_sum",
                    ctx=ast.Load(),
                ),
                args=[ast.Name(id=self.name, ctx=ast.Load())],
                keywords=[],
            ),
        )

    def to_sampler(self, graph):
        args = [ast.Name(id="rng_key", ctx=ast.Load())]

        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=self.distribution,
                    attr="forward",
                    ctx=ast.Load(),
                ),
                args=args,
                keywords=[],
            ),
        )


class Var(object):
    def __init__(
        self,
        name: str,
        value: Union[ast.Constant, ast.Num],
        is_returned: bool,
    ):
        self.name = name
        self.value = value
        self.is_returned = is_returned

    def to_logpdf(self):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.value
        )

    def to_logpdf_iadd(self):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.value
        )

    def to_sampler(self, graph):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.value
        )


class Transformation(object):
    def __init__(
        self,
        name: str,
        expression: Optional[Union[int, float]],
        args: Optional[List[Union[int, float]]],
        is_returned: bool,
    ):
        self.name = name
        self.expression = expression
        self.args = args
        self.is_returned = is_returned

    def __str__(self):
        return "{} = {}".format(self.name, astor.code_gen.to_source(self.expression))

    def to_logpdf(self):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.expression
        )

    def to_logpdf_iadd(self):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.expression
        )

    def to_sampler(self, graph):
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())], value=self.expression
        )
