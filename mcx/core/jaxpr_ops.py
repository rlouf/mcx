"""A collection of operations/transformations on JAX expressions.
"""
import jax.core
import jax.lax
from jax.util import safe_map

from functools import wraps
from typing import List, Dict, Tuple, Any

Array = Any


def jax_lax_identity(x: Array) -> Array:
    """Identity operator.

    Intrinsingly, it seems jax.lax does not have a public identity operation?
    """
    return x


def jaxpr_find_constvars(
    jaxpr: jax.core.Jaxpr, consts: List[jax.core.Var]
) -> List[jax.core.Var]:
    """Find all intermediates variables in a JAX expression which are expected to be constants.

    Parameters
    ----------
    jaxpr: JAX expression.
    consts: List of known constant variables in the JAX expression.

    Returns
    -------
    List of all intermediate constant variables.
    """
    constvars_dict = {str(v): v for v in consts}
    for eqn in jaxpr.eqns:
        # Are inputs literal or const variables?
        is_const_invars = [
            str(v) in constvars_dict or type(v) is jax.core.Literal for v in eqn.invars
        ]
        if all(is_const_invars):
            constvars_dict.update({str(v): v for v in eqn.outvars})
    return list(constvars_dict.values())


def jaxpr_find_denormalize_mapping(
    jaxpr: jax.core.Jaxpr, consts: List[jax.core.Var]
) -> Dict[jax.core.Var, Tuple[jax.core.Primitive, jax.core.Var]]:
    """Find all assignment simplifications in a JAX expression when denormalizing.

    More specifically, this method is looking to simplify `add` and `sub` operations, with output linear
    with respect to the Jaxpr outputs, and where one of the input is constant. It returns the simplified mapping
    between input and output of `add`/`sub` ops which can be removed.

    Parameters
    ----------
    jaxpr: JAX expression.
    consts: List of known constant variables in the JAX expression.

    Returns
    -------
    Simplified mapping between `add` output and input (with the proper assignment lax op `identity` or `neg`).
    """
    denormalize_mapping = {}
    # List of linear ops which can be traversed backward from the outputs.
    denorm_supported_linear_ops = [
        jax.lax.broadcast_in_dim_p,
        jax.lax.broadcast_p,
        jax.lax.neg_p,
        jax.lax.reshape_p,
        jax.lax.squeeze_p,
    ]
    # Collection of variables linear with respect to the Jaxpr final outputs.
    linear_vars = set(jaxpr.outvars)

    # Traversing backward the graph of operations.
    for eqn in jaxpr.eqns[::-1]:
        if eqn.primitive in denorm_supported_linear_ops:
            # Can continue denormalizing inputs if all outputs are in the linear vars collection.
            if all([o in linear_vars for o in eqn.outvars]):
                linear_vars |= set(eqn.invars)
        elif eqn.primitive == jax.lax.add_p and eqn.outvars[0] in linear_vars:
            lhs_invar, rhs_invar = eqn.invars[0], eqn.invars[1]
            # Mapping the output to the non-const input.
            if lhs_invar in consts or type(lhs_invar) is jax.core.Literal:
                linear_vars.add(rhs_invar)
                denormalize_mapping[eqn.outvars[0]] = (jax_lax_identity, rhs_invar)
            elif rhs_invar in consts or type(rhs_invar) is jax.core.Literal:
                linear_vars.add(lhs_invar)
                denormalize_mapping[eqn.outvars[0]] = (jax_lax_identity, lhs_invar)
        elif eqn.primitive == jax.lax.sub_p and eqn.outvars[0] in linear_vars:
            lhs_invar, rhs_invar = eqn.invars[0], eqn.invars[1]
            # Mapping the output to the non-const input (or the negative).
            if lhs_invar in consts or type(lhs_invar) is jax.core.Literal:
                linear_vars.add(rhs_invar)
                denormalize_mapping[eqn.outvars[0]] = (jax.lax.neg, rhs_invar)
            elif rhs_invar in consts or type(rhs_invar) is jax.core.Literal:
                linear_vars.add(lhs_invar)
                denormalize_mapping[eqn.outvars[0]] = (jax_lax_identity, lhs_invar)

    return denormalize_mapping


def jaxpr_denormalize(jaxpr, consts, *args):
    """Denormalize a Jaxpr, i.e. removing any normalizing constant added to the output.

    This method is analysing the Jaxpr graph, simplifying it by skipping any unnecessary constant
    addition, and then it runs the method step-by-step to get the output values.

    Parameters
    ----------
    jaxpr: JAX expression.
    consts: Values assigned to the Jaxpr constant variables.
    args: Input values to the method.

    Returns
    -------
    Output values of the denormalized logpdf.
    """
    # Denormalized simplification mapping.
    denorm_mapping = jaxpr_find_denormalize_mapping(jaxpr, jaxpr.constvars)
    # Mapping from variable -> value
    env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is jax.core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    # Bind args and consts to environment
    write(jax.core.unitvar, jax.core.unit)
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Similar to a classic eval Jaxpr loop, just skipping the op with mapping available
    for eqn in jaxpr.eqns:
        if len(eqn.outvars) == 1 and eqn.outvars[0] in denorm_mapping:
            # Output registered: skip the primitive and map directly to one of the input.
            outvar = eqn.outvars[0]
            map_primitive, map_invar = (
                denorm_mapping[outvar][0],
                denorm_mapping[outvar][1],
            )
            # Mapping the inval to output var (identity or neg).
            inval = read(map_invar)
            outval = map_primitive(inval)
            write(outvar, outval)
        else:
            # Usual map: calling the primitive and mapping the output values.
            invals = safe_map(read, eqn.invars)
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(write, eqn.outvars, outvals)
    # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars)


def denormalize(logpdf_fn):
    """Denormalizing decorator for MCX logpdfs.

    The method returned by the `denormalize` decorator is a simplification of the Jaxpr,
    removing any constant in the output logpdf.
    """

    @wraps(logpdf_fn)
    def wrapped(*args, **kwargs):
        # TODO: flattening/unflattening of inputs/outputs?
        closed_jaxpr = jax.make_jaxpr(logpdf_fn)(*args, **kwargs)
        out = jaxpr_denormalize(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        # Assuming a single output at the moment?
        return out[0]

    return wrapped
