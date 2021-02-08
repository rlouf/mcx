"""A collection of operations/transformations on JAX expressions.
"""
import jax.core
import jax.lax
from jax.util import safe_map

import enum
from functools import wraps
from typing import List, Dict, Tuple, Any, Type, TypeVar, Callable

Array = Any
"""Generic Array type.
"""
TState = TypeVar("TState")
"""Generic Jaxpr visitor state.
"""


jaxpr_high_order_primitives_to_subjaxprs = {
    jax.lax.cond_p: lambda jxpr: jxpr.params["branches"],
    jax.lax.while_p: lambda jxpr: (
        jxpr.params["cond_jaxpr"],
        jxpr.params["body_jaxpr"],
    ),
    jax.lax.scan_p: lambda jxpr: (jxpr.params["jaxpr"],),
    jax.core.CallPrimitive: lambda jxpr: (
        jxpr.params["call_jaxpr"],
    ),  # xla_call, from jax.jit
    jax.core.MapPrimitive: lambda jxpr: (jxpr.params["call_jaxpr"],),
}
"""Collection of high-order Jax primitives, with sub-Jaxprs.
"""
jaxpr_high_order_primitives = set(jaxpr_high_order_primitives_to_subjaxprs.keys())


def jax_lax_identity(x: Array) -> Array:
    """Identity operator.

    Intrinsingly, it seems jax.lax does not have a public identity operation?
    """
    return x


def jax_is_high_order_primitive(
    eqn: jax.core.JaxprEqn,
) -> bool:
    """Is the input Jaxpr equation corresponding to a high-order Jax primitive?

    Parameters
    ----------
    Returns
    -------
    """
    is_high_order = (eqn.primitive in jaxpr_high_order_primitives_to_subjaxprs) or (
        type(eqn.primitive) in jaxpr_high_order_primitives_to_subjaxprs
    )
    return is_high_order


def jaxpr_find_sub_jaxprs(
    eqn: jax.core.JaxprEqn,
) -> List[jax.core.Jaxpr]:
    """Is the input Jaxpr equation corresponding to a high-order Jax primitive?

    Parameters
    ----------
    Returns
    -------
    """
    primitive_type = type(eqn.primitive)
    if eqn.primitive in jaxpr_high_order_primitives_to_subjaxprs:
        return jaxpr_high_order_primitives_to_subjaxprs[eqn.primitive](eqn)
    elif primitive_type in jaxpr_high_order_primitives_to_subjaxprs:
        return jaxpr_high_order_primitives_to_subjaxprs[primitive_type](eqn)
    return []


def jaxpr_visitor(
    jaxpr: jax.core.Jaxpr,
    initial_state: TState,
    visitor_fn: Callable[[jax.core.JaxprEqn, TState], TState],
    map_sub_states_fn: Callable[[jax.core.JaxprEqn, TState], List[TState]],
    reduce_sub_states_fn: Callable[[jax.core.JaxprEqn, TState, List[TState]], TState],
    reverse: bool = False,
) -> Tuple[TState, List[Any]]:
    """Visitor pattern on a Jaxpr, traversing equations and supporting higher-order primitives
    with sub-Jaxprs.

    Parameters
    ----------
    initial_state: Initial state to feed to the visitor method.
    visitor_fn: Visitor function, taking an input state and Jaxpr, outputting an updated state.
    init_sub_states_fn: Initializing method for higher-order primitives sub-Jaxprs. Taking as input
        the existing state, and outputting input states to respective sub-Jaxprs.
    reverse: Traverse the Jaxpr equations in reverse order.

    Returns
    -------
    Output state of the last iteration.
    """
    state = initial_state
    subjaxprs_visit = []

    equations = jaxpr.eqns if not reverse else jaxpr.eqns[::-1]
    for eqn in equations:
        if jax_is_high_order_primitive(eqn):
            init_sub_states = map_sub_states_fn(eqn, state)
            sub_jaxprs = jaxpr_find_sub_jaxprs(eqn)
            # Map visitor method to each sub-jaxpr.
            res_sub_states = [
                jaxpr_visitor(
                    sub_jaxpr,
                    sub_state,
                    visitor_fn,
                    map_sub_states_fn,
                    reduce_sub_states_fn,
                    reverse,
                )
                for sub_jaxpr, sub_state in zip(sub_jaxprs, init_sub_states)
            ]
            # Reduce to update the current state.
            state = reduce_sub_states_fn(eqn, state, [v[0] for v in res_sub_states])
            subjaxprs_visit.append(res_sub_states)
        else:
            # Common Jaxpr equation: apply the visitor and update state.
            state = visitor_fn(eqn, state)
            subjaxprs_visit.append(None)
    return state, subjaxprs_visit


class ConstVarStatus(enum.Enum):
    """Const variable status.

    For the application to constant simplification in logpdfs, we do not need a full constant
    folding in the graph of operations (which can be quite expensive, in computation and memory), but
    just to keep track of constant variables and whether these are non-finite or not.
    """

    Unknown = 0
    NonFinite = 1


ConstVarState = Dict[jax.core.Var, ConstVarStatus]
"""Const variables visitor state: dictionary associating const variables with their status.
"""


def jaxpr_find_constvars_visitor_fn(
    eqn: jax.core.JaxprEqn,
    state: ConstVarState,
) -> ConstVarState:
    """Jaxpr find const variables visitor method: propagating constant through ops.

    This method is implementing a very simple logic, assuming that as method in Jax should be pure
    functions, any output of a function with constant inputs is constant.

    Parameters
    ----------
    eqn: Jaxpr equation.
    state: Current collection of constant variables.

    Returns
    -------
    Updated constant variables collection with outputs of the Jaxpr equation.
    """
    # Common ops logic: are inputs literal or const variables?
    # NOTE: Jax literal are not hashable!
    is_const_invars = [type(v) is jax.core.Literal or v in state for v in eqn.invars]
    if all(is_const_invars):
        state.update({v: ConstVarStatus.Unknown for v in eqn.outvars})
    return state


def jaxpr_find_constvars_map_sub_states_fn(
    eqn: jax.core.JaxprEqn, state: ConstVarState
) -> List[ConstVarState]:
    """Map the constant variables collection to sub-jaxprs initial constant collections.

    The method is performing a simple mapping of constant variables of the main jaxpr to the inputs
    of the sub-jaxprs.

    Parameters
    ----------
    eqn: Jaxpr equation with high order primitive (xla_call, ...).
    state: Constant variables collection.

    Returns
    -------
    List of initial const variable states corresponding to each sub-jaxpr.
    """
    # Mapping the current state to the sub-jaxprs.
    primitive_type = type(eqn.primitive)
    sub_jaxprs = jaxpr_find_sub_jaxprs(eqn)

    if primitive_type == jax.core.CallPrimitive:
        # Jit compiled sub-jaxpr: map eqn inputs to sub-jaxpr inputs.
        sub_init_state = {}
        for eqn_invar, sub_invar in zip(eqn.invars, sub_jaxprs[0].invars):
            # Add a constant variables if marked constant in the sub-jaxpr.
            if eqn_invar in state or type(sub_invar) is jax.core.Literal:
                sub_init_state[sub_invar] = False
        return [sub_init_state]
    else:
        # TODO: support other high primitives. No constants passed at the moment.
        sub_init_states = [{} for _ in range(len(sub_jaxprs))]
        return sub_init_states


def jaxpr_find_constvars_reduce_sub_states_fn(
    eqn: jax.core.JaxprEqn, state: ConstVarState, sub_states: List[ConstVarState]
) -> ConstVarState:
    """Reduce the collection of sub-jaxpr const variables states to update to main jaxpr state.

    The method is performing a simple update of the main jaxpr state using the result of the
    sub-jaxprs (i.e. whether the latter are constants).

    Parameters
    ----------
    eqn: Main jaxpr equation.
    state: Main jaxpr current const variables state.
    sub_states: Sub-jaxprs final const variables states.

    Returns
    -------
    Updated main Jaxpr constant variables state.
    """
    primitive_type = type(eqn.primitive)
    if primitive_type == jax.core.CallPrimitive:
        # Jit compiled sub-jaxpr.
        sub_jaxpr = eqn.params["call_jaxpr"]
        sub_state = sub_states[0]
        for eqn_outvar, sub_outvar in zip(eqn.outvars, sub_jaxpr.outvars):
            # Add a constant variables if marked constant in the sub-jaxpr.
            if sub_outvar in sub_state:
                state[eqn_outvar] = sub_state[sub_outvar]
    else:
        # TODO: support other high primitives. No constants passed at the moment.
        pass
    return state


def jaxpr_find_constvars(
    jaxpr: jax.core.Jaxpr, consts: List[jax.core.Var]
) -> Dict[jax.core.Var, ConstVarStatus]:
    """Find all intermediates variables in a JAX expression which are expected to be constants.

    Parameters
    ----------
    jaxpr: JAX expression.
    consts: List of known constant variables in the JAX expression.

    Returns
    -------
    List of all intermediate constant variables.
    """
    # Start with the list of input constants.
    const_state = {c: False for c in consts}
    const_state, _ = jaxpr_visitor(
        jaxpr,
        const_state,
        jaxpr_find_constvars_visitor_fn,
        jaxpr_find_constvars_map_sub_states_fn,
        jaxpr_find_constvars_reduce_sub_states_fn,
        reverse=False,
    )
    return const_state


def jaxpr_find_constvars_old(
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
        jax.lax.reduce_sum_p,
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
