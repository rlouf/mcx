import jax.core
import jax.lax

from typing import Dict, Any
from jax.core import Jaxpr, Literal, Var, unitvar, unit, extract_call_jaxpr
from jax.util import (
    safe_zip,
    safe_map,
    partial,
    curry,
    prod,
    partialmethod,
    tuple_insert,
    tuple_delete,
)
import jax.linear_util as lu
from jax._src import source_info_util


def eval_jaxpr(jaxpr: Jaxpr, consts, *args):
    def read(v):
        if type(v) is Literal:
            return v.val
        else:
            return env[v]

    def write(v, val):
        env[v] = val

    env: Dict[Var, Any] = {}
    write(unitvar, unit)
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.invars)
        call_jaxpr, params = extract_call_jaxpr(eqn.primitive, eqn.params)
        if call_jaxpr:
            subfuns = [lu.wrap_init(partial(eval_jaxpr, call_jaxpr, ()))]
        else:
            subfuns = []
        with source_info_util.user_context(eqn.source_info):
            ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
        if eqn.primitive.multiple_results:
            map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    return map(read, jaxpr.outvars)
