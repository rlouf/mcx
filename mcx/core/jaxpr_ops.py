"""A collection of operations/transformations on JAX expressions.
"""
import jax.core
import jax.lax
from jax.util import safe_map
import numpy as np

import copy
import enum

from dataclasses import dataclass
from functools import wraps
from typing import List, Dict, Optional, Set, Tuple, Any, Type, TypeVar, Callable

Array = Any
"""Generic Array type.
"""
TState = TypeVar("TState")
"""Generic Jaxpr visitor state.
"""
TRecState = Tuple[TState, List[Optional[List["TRecState"]]]]
"""Full recursive state, representing the visitor state of the Jaxpr as well as
the sub-states of all sub-jaxprs.
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
) -> TRecState:
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


@dataclass
class ConstVarInfo:
    """Const variable additional information.

    For the application of constant simplification in logpdfs, we do not need a full constant
    folding in the graph of operations (which can be quite expensive, in computation and memory), but
    just to keep track of constant variables, and some additional information:

    Parameters
    ----------
    is_non_finite: whether the constant is non-finite. false by default.
    is_uniform: whether the constant is a uniform tensor. false by default.
    """

    is_non_finite: bool = False
    is_uniform: bool = False


ConstVarState = Dict[jax.core.Var, ConstVarInfo]
"""Const variables visitor state: dictionary associating const variables with their info.
"""
ConstVarRecState = Tuple[ConstVarState, List[Optional[List["ConstVarRecState"]]]]


# Garthoks = Union[Garthok, Iterable['Garthoks']]


def get_variable_const_info(v: Any, state: ConstVarState) -> ConstVarInfo:
    """Get the constant info on a variable (or literal).

    Parameters
    ----------
    v: Input variable or literal
    state: Constant variable state.

    Returns
    -------
    Optional const info. None if not a constant variable.
    """
    if type(v) is jax.core.Literal:
        return ConstVarInfo(is_non_finite=not bool(np.isfinite(v.val)), is_uniform=True)
    return state.get(v, None)


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
    invars_const_info = [get_variable_const_info(v, state) for v in eqn.invars]
    if all(is_const_invars):
        # Using a form of heuristic here: outputs are non-finite if one the input is.
        # TODO: refine this logic per op.
        is_non_finite_outvars = any(
            [s is not None and s.is_non_finite for s in invars_const_info]
        )
        # Another heuristic to refine! if all inputs are uniform, all outputs are uniform.
        # TODO: refine the logic per op supported.
        is_uniform_outvars = all(
            [s is not None and s.is_uniform for s in invars_const_info]
        )

        outvar_const_info = ConstVarInfo(
            is_non_finite=is_non_finite_outvars, is_uniform=is_uniform_outvars
        )
        state.update({v: copy.copy(outvar_const_info) for v in eqn.outvars})
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
            if eqn_invar in state:
                # Add a constant variables if marked constant in the sub-jaxpr.
                sub_init_state[sub_invar] = state[eqn_invar]
            elif type(sub_invar) is jax.core.Literal:
                # Literal argument: check the value fo the status.
                sub_init_state[sub_invar] = get_variable_const_info(sub_invar, None)
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
    jaxpr: jax.core.Jaxpr, constvars: Dict[jax.core.Var, ConstVarInfo]
) -> ConstVarRecState:
    """Find all intermediates variables in a JAX expression which are expected to be constants.

    Parameters
    ----------
    jaxpr: JAX expression.
    constvars: List of known constant variables in the JAX expression.

    Returns
    -------
    List of all intermediate constant variables.
    """
    # Start with the collection of input constants.
    const_state = copy.copy(constvars)
    const_rec_state = jaxpr_visitor(
        jaxpr,
        const_state,
        jaxpr_find_constvars_visitor_fn,
        jaxpr_find_constvars_map_sub_states_fn,
        jaxpr_find_constvars_reduce_sub_states_fn,
        reverse=False,
    )
    return const_rec_state


DenormMapState = Tuple[
    Dict[jax.core.Var, Tuple[Any, jax.core.Var]], Set[jax.core.Var], ConstVarRecState
]
"""Denormalization state, combination of:
    - dictionary of variable mapping, corresponding to `add` or `sub` ops which can be simplified;
    - set of variables which can be traverse backward for denormalization;
    - full recursive const variable state of the Jaxpr.
"""
DenormMapRecState = Tuple[DenormMapState, List[Optional[List["DenormMapRecState"]]]]


def jax_is_literal(v: Any) -> bool:
    """Is the input variable a Jax core literal?"""
    return type(v) is jax.core.Literal


def jax_is_non_finite_constant(v: Any, const_state: ConstVarState):
    """"""
    return (jax_is_literal(v) and not np.isfinite(v.val)) or (
        v in const_state and const_state[v].is_non_finite
    )


def jaxpr_denorm_propagate_blocking_eqn(
    eqn: jax.core.JaxprEqn, state: DenormMapState
) -> Tuple[Set[jax.core.Var], Set[jax.core.Var]]:
    """Default primitive propagation: blocking the denormalization of input variables."""
    invalid_invars = {v for v in eqn.invars if not jax_is_literal(v)}
    return set(), invalid_invars


def jaxpr_denorm_propagate_linear_eqn(
    eqn: jax.core.JaxprEqn, state: DenormMapState
) -> Tuple[Set[jax.core.Var], Set[jax.core.Var]]:
    """fdsfafasd

    fasdfasd
    """
    # Base check for back-propagate through a valid op: all outputs must be valid denorm variables.
    invars = {v for v in eqn.invars if not jax_is_literal(v)}
    _, denorm_valid_vars, _ = state
    check_denorm_propagate = all([o in denorm_valid_vars for o in eqn.outvars])
    if all([o in denorm_valid_vars for o in eqn.outvars]):
        return invars, set()
    # Default case: blocking back-propagation of denormalization.
    return set(), invars


def jaxpr_denorm_propagate_mul_eqn(
    eqn: jax.core.JaxprEqn, state: DenormMapState
) -> Tuple[Set[jax.core.Var], Set[jax.core.Var]]:
    """fdsfafasd

    fasdfasd
    """
    _, denorm_valid_vars, constvar_full_state = state
    constvar_state, _ = constvar_full_state

    def is_var_constant(v: Any) -> bool:
        return type(v) is jax.core.Literal or v in constvar_state

    invars = {v for v in eqn.invars if not jax_is_literal(v)}
    # Propagate denormalization if one of the input is a uniform constant.
    all_valid_outvars = all([o in denorm_valid_vars for o in eqn.outvars])
    any_invar_const = is_var_constant(eqn.invars[0]) or is_var_constant(eqn.invars[1])
    if all_valid_outvars and any_invar_const:
        return invars, set()
    # Default case: blocking back-propagation of denormalization.
    return set(), invars


def jaxpr_denorm_propagate_div_eqn(
    eqn: jax.core.JaxprEqn, state: DenormMapState
) -> Tuple[Set[jax.core.Var], Set[jax.core.Var]]:
    """fdsfafasd

    fasdfasd
    """
    _, denorm_valid_vars, constvar_full_state = state
    constvar_state, _ = constvar_full_state

    def is_var_constant(v: Any) -> bool:
        return type(v) is jax.core.Literal or v in constvar_state

    invars = {v for v in eqn.invars if not jax_is_literal(v)}
    # Propagate denormalization if second input is a uniform constant.
    all_valid_outvars = all([o in denorm_valid_vars for o in eqn.outvars])
    second_invar_const = is_var_constant(eqn.invars[1])
    if all_valid_outvars and second_invar_const:
        return invars, set()
    # Default case: blocking back-propagation of denormalization.
    return set(), invars


def jaxpr_denorm_propagate_select_eqn(
    eqn: jax.core.JaxprEqn, state: DenormMapState
) -> Tuple[Set[jax.core.Var], Set[jax.core.Var]]:
    """fdsfafasd

    fasdfasd
    """
    _, denorm_valid_vars, constvar_full_state = state
    constvar_state, _ = constvar_full_state
    invar_pred, invar_true, invar_false = eqn.invars
    all_valid_outvars = all([o in denorm_valid_vars for o in eqn.outvars])

    if jax_is_non_finite_constant(invar_true, constvar_state) and all_valid_outvars:
        valid_vars = set() if jax_is_literal(invar_false) else {invar_false}
        invalid_vars = set() if jax_is_literal(invar_pred) else {invar_pred}
        return valid_vars, invalid_vars

    if jax_is_non_finite_constant(invar_false, constvar_state) and all_valid_outvars:
        valid_vars = set() if jax_is_literal(invar_true) else {invar_true}
        invalid_vars = set() if jax_is_literal(invar_pred) else {invar_pred}
        return valid_vars, invalid_vars

    # Default case: blocking back-propagation of denormalization.
    invars = {v for v in eqn.invars if not jax_is_literal(v)}
    return set(), invars


jaxpr_eqn_denorm_propagate_rules = {
    jax.lax.broadcast_in_dim_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.broadcast_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.neg_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.reshape_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.squeeze_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.reduce_sum_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.add_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.sub_p: jaxpr_denorm_propagate_linear_eqn,
    jax.lax.mul_p: jaxpr_denorm_propagate_mul_eqn,
    jax.lax.div_p: jaxpr_denorm_propagate_div_eqn,
    jax.lax.select_p: jaxpr_denorm_propagate_select_eqn,
}
"""
"""


def jaxpr_denorm_mapping_visitor_fn(
    eqn: jax.core.JaxprEqn,
    state: DenormMapState,
) -> DenormMapState:
    """pass
    fdsafas
    """
    # Un-stack input complex input state!
    denorm_map_dict, denorm_valid_vars, constvar_full_state = state
    constvar_state, constvar_sub_states = constvar_full_state

    def is_var_constant(v: Any) -> bool:
        return type(v) is jax.core.Literal or v in constvar_state

    def denorm_linear_op(invar, outvar, replace_op):
        denorm_valid_vars.add(invar)
        denorm_map_dict[outvar] = (replace_op, invar)

    # Check which input variables to keep propagating the denormalization.
    eqn_propagate_check_fn = jaxpr_eqn_denorm_propagate_rules.get(
        eqn.primitive, jaxpr_denorm_propagate_blocking_eqn
    )
    valid_invars, invalid_invars = eqn_propagate_check_fn(eqn, state)
    # Update the global denorm valid vars accordingly.
    denorm_valid_vars |= valid_invars
    denorm_valid_vars -= invalid_invars

    # Add and sub operations which can be simplified.
    if eqn.primitive == jax.lax.add_p and eqn.outvars[0] in denorm_valid_vars:
        lhs_invar, rhs_invar = eqn.invars[0], eqn.invars[1]
        # Mapping the output to the non-const input.
        if is_var_constant(lhs_invar):
            denorm_linear_op(rhs_invar, eqn.outvars[0], jax_lax_identity)
        elif is_var_constant(rhs_invar):
            denorm_linear_op(lhs_invar, eqn.outvars[0], jax_lax_identity)
    elif eqn.primitive == jax.lax.sub_p and eqn.outvars[0] in denorm_valid_vars:
        lhs_invar, rhs_invar = eqn.invars[0], eqn.invars[1]
        # Mapping the output to the non-const input (or the negative).
        if is_var_constant(lhs_invar):
            denorm_linear_op(rhs_invar, eqn.outvars[0], jax.lax.neg)
        elif is_var_constant(rhs_invar):
            denorm_linear_op(lhs_invar, eqn.outvars[0], jax_lax_identity)

    # Update the constvar sub-states list, to keep sync. with equations in the jaxpr.
    constvar_sub_states = constvar_sub_states[:-1]
    constvar_full_state = constvar_state, constvar_sub_states
    # Re-construct updated state.
    return (denorm_map_dict, denorm_valid_vars, constvar_full_state)


def jaxpr_denorm_mapping_map_sub_states_fn(
    eqn: jax.core.JaxprEqn, state: DenormMapState
) -> List[DenormMapState]:
    """"""
    denorm_map_dict, denorm_valid_vars, constvar_full_state = state
    constvar_state, constvar_sub_states = constvar_full_state

    sub_jaxprs = jaxpr_find_sub_jaxprs(eqn)
    assert len(constvar_sub_states[-1]) == len(sub_jaxprs)

    primitive_type = type(eqn.primitive)
    if primitive_type == jax.core.CallPrimitive:
        # Jit compiled sub-jaxpr: map eqn outputs to sub-jaxpr outputs.
        sub_jaxpr, sub_const_state = sub_jaxprs[0], constvar_sub_states[-1][0]
        # Map the denorm valid vars to the output of the sub-jaxprs.
        denorm_sub_valid_vars = {
            sub_outvar
            for eqn_outvar, sub_outvar in zip(eqn.outvars, sub_jaxpr.outvars)
            if eqn_outvar in denorm_valid_vars
        }
        denorm_sub_state = ({}, denorm_sub_valid_vars, sub_const_state)
        return [denorm_sub_state]
    else:
        # TODO: support other high primitives. No constants passed at the moment.
        denorm_sub_states = [state for _ in sub_jaxprs]
        return denorm_sub_states


def jaxpr_denorm_mapping_reduce_sub_states_fn(
    eqn: jax.core.JaxprEqn, state: DenormMapState, sub_states: List[DenormMapState]
) -> DenormMapState:
    """"""
    sub_jaxprs = jaxpr_find_sub_jaxprs(eqn)
    assert len(sub_states) == len(sub_jaxprs)

    denorm_map_dict, denorm_valid_vars, constvar_full_state = state
    primitive_type = type(eqn.primitive)
    if primitive_type == jax.core.CallPrimitive:
        # Jit compiled sub-jaxpr: map valid sub-jaxpr inputs to update denorm valid variables.
        sub_jaxpr = sub_jaxprs[0]
        _, sub_denorm_valid_vars, _ = sub_states[0]
        denorm_valid_vars |= {
            eqn_invar
            for eqn_invar, sub_invar in zip(eqn.invars, sub_jaxpr.invars)
            if sub_invar in sub_denorm_valid_vars
        }

    # Update the constvar sub-states list, to keep sync. with equations in the jaxpr.
    constvar_state, constvar_sub_states = constvar_full_state
    constvar_full_state = constvar_state, constvar_sub_states[:-1]
    # TODO: fix properly.
    state = denorm_map_dict, denorm_valid_vars, constvar_full_state
    return state


def jaxpr_find_denormalize_mapping(
    jaxpr: jax.core.Jaxpr, constvar_state: ConstVarRecState
) -> DenormMapRecState:
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
    # Initialize the denormalize state, starting from the ouput variables.
    denormalize_mapping = {}
    denorm_valid_vars = set(jaxpr.outvars)
    denorm_state = (denormalize_mapping, denorm_valid_vars, constvar_state)
    # NOTE: scanning the jaxpr in reverse order.
    denorm_rec_state = jaxpr_visitor(
        jaxpr,
        denorm_state,
        jaxpr_denorm_mapping_visitor_fn,
        jaxpr_denorm_mapping_map_sub_states_fn,
        jaxpr_denorm_mapping_reduce_sub_states_fn,
        reverse=True,
    )
    return denorm_rec_state


DenormRunState = Tuple[Dict[jax.core.Var, Any], DenormMapRecState]
"""Denormalization run state.
"""
DenormRunRecState = Tuple[DenormRunState, List[Optional[List["DenormRunRecState"]]]]


def jaxpr_denorm_run_visitor_fn(
    eqn: jax.core.JaxprEqn,
    state: DenormRunState,
) -> DenormRunState:
    """pass
    fdsafas
    """
    denorm_env, denorm_map_rec_state = state
    denorm_mapping, _, _ = denorm_map_rec_state[0]

    def read_env(var):
        if type(var) is jax.core.Literal:
            return var.val
        return denorm_env[var]

    def write_env(var, val):
        denorm_env[var] = val

    if len(eqn.outvars) == 1 and eqn.outvars[0] in denorm_mapping:
        # Output registered: skip the primitive and map directly to one of the input.
        outvar = eqn.outvars[0]
        outvar_mapping = denorm_mapping[outvar]
        map_primitive, map_invar = (outvar_mapping[0], outvar_mapping[1])
        print(denorm_mapping, map_primitive, map_invar)
        # Mapping the inval to output var (identity or neg).
        inval = read_env(map_invar)
        outval = map_primitive(inval)
        write_env(outvar, outval)
    else:
        # Usual map: calling the primitive and mapping the output values.
        invals = safe_map(read_env, eqn.invars)
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        safe_map(write_env, eqn.outvars, outvals)

    # Pop first element in the recursive denorm state, to keep in sync.
    denorm_map_sub_states = denorm_map_rec_state[1][1:]
    denorm_map_rec_state = (denorm_map_rec_state[0], denorm_map_sub_states)
    # Returning updated environment.
    return denorm_env, denorm_map_rec_state


def jaxpr_denorm_run_map_sub_states_fn(
    eqn: jax.core.JaxprEqn, state: DenormRunState
) -> List[DenormRunState]:
    """"""
    # denorm_map_dict, denorm_valid_vars, constvar_full_state = state
    # constvar_state, constvar_sub_states = constvar_full_state


def jaxpr_denorm_run_reduce_sub_states_fn(
    eqn: jax.core.JaxprEqn, state: DenormRunState, sub_states: List[DenormRunState]
) -> DenormRunState:
    """"""
    # sub_jaxprs = jaxpr_find_sub_jaxprs(eqn)
    # assert len(sub_states) == len(sub_jaxprs)


def jaxpr_denormalize_run(jaxpr: jax.core.Jaxpr, consts, *args) -> DenormRunRecState:
    """TODO

    Parameters
    ----------
    jaxpr: JAX expression.
    consts: List of known constant variables in the JAX expression.

    Returns
    -------
    Simplified mapping between `add` output and input (with the proper assignment lax op `identity` or `neg`).
    """
    # Generate the denormalization simplifying mapping.
    constvars = {v: ConstVarInfo(False, True) for v in jaxpr.constvars}
    constvar_full_state = jaxpr_find_constvars(jaxpr, constvars)
    denorm_map_state = jaxpr_find_denormalize_mapping(jaxpr, constvar_full_state)

    # Initialize the denormalize env state, starting from the input variables.
    denormalize_env = {}

    def write_env(var, val):
        denormalize_env[var] = val

    # Bind args and consts to denormalization environment.
    write_env(jax.core.unitvar, jax.core.unit)
    safe_map(write_env, jaxpr.invars, args)
    safe_map(write_env, jaxpr.constvars, consts)

    print(denormalize_env)
    print(jaxpr)
    print(denorm_map_state)

    denorm_init_state = (denormalize_env, denorm_map_state)
    # NOTE: scanning the jaxpr in reverse order.
    denorm_run_state = jaxpr_visitor(
        jaxpr,
        denorm_init_state,
        jaxpr_denorm_run_visitor_fn,
        jaxpr_denorm_run_map_sub_states_fn,
        jaxpr_denorm_run_reduce_sub_states_fn,
        reverse=False,
    )
    denorm_outenv = denorm_run_state[0][0]
    outvals = [denorm_outenv[v] for v in jaxpr.outvars]
    return outvals


def jaxpr_denormalize_old(jaxpr, consts, *args):
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
        out = jaxpr_denormalize_run(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        # Assuming a single output at the moment?
        return out[0]

    return wrapped
