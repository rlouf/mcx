from typing import TypedDict
import jax
import jax.lax
import numpy as onp
import pytest
import jax.numpy as jnp
import numpy as np

from mcx.core.jaxpr_ops import (
    jax_lax_identity,
    jaxpr_find_constvars,
    jaxpr_find_denormalize_mapping,
    denormalize,
    ConstVarInfo,
)


find_constvars_test_functions = [
    # Simple constant propagation.
    {
        "fn": lambda x: x + np.ones((2,)) + np.exp(2.0),
        "info": ConstVarInfo(False, True),
    },
    # TODO: Simple constant propagation, non-uniform.
    # {
    #     "fn": lambda x: x + jnp.array([1.0, 2.0]) + np.exp(2.0),
    #     "info": ConstVarInfo(False, False),
    # },
    # Handle properly jax.jit sub-jaxpr.
    {
        "fn": lambda x: jax.jit(lambda y: y + jnp.ones((2,)))(x) + np.exp(2.0),
        "info": ConstVarInfo(False, True),
    },
    # Simple inf constant propagation.
    {"fn": lambda x: x + np.ones((2,)) + np.inf, "info": ConstVarInfo(True, True)},
    # Handle properly jax.jit sub-jaxpr + inf constant.
    {
        "fn": lambda x: jax.jit(lambda y: y + jnp.full((2,), np.inf))(x) + np.exp(2.0),
        "info": ConstVarInfo(True, True),
    },
    # TODO: test pmap, while, scan, cond.
]


@pytest.mark.parametrize("case", find_constvars_test_functions)
def test__jaxpr_find_constvars__propagate_constants(case):
    test_fn = case["fn"]
    expected_const_info = case["info"]
    typed_jaxpr = jax.make_jaxpr(test_fn)(1.0)

    print(typed_jaxpr.consts, typed_jaxpr.jaxpr.constvars)

    # All inputs consts, outputs should be consts!
    constvars = {v: ConstVarInfo(False, True) for v in typed_jaxpr.jaxpr.invars}
    constvars.update(
        {v: ConstVarInfo(False, True) for v in typed_jaxpr.jaxpr.constvars}
    )

    constvars = jaxpr_find_constvars(typed_jaxpr.jaxpr, constvars).state
    for outvar in typed_jaxpr.jaxpr.outvars:
        assert outvar in constvars
        assert constvars[outvar] == expected_const_info


denorm_expected_add_mapping_op = [
    {"fn": lambda x: x + 1.0, "expected_op": jax_lax_identity},
    {"fn": lambda x: 1.0 + x, "expected_op": jax_lax_identity},
    {"fn": lambda x: x - 1.0, "expected_op": jax_lax_identity},
    {"fn": lambda x: 1.0 - x, "expected_op": jax.lax.neg},
]


@pytest.mark.parametrize("case", denorm_expected_add_mapping_op)
def test__jaxpr_find_denormalize_mapping__add_sub__proper_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    constvars = {v: ConstVarInfo(False, True) for v in typed_jaxpr.jaxpr.constvars}
    constvar_state = jaxpr_find_constvars(typed_jaxpr.jaxpr, constvars)

    denorm_rec_state = jaxpr_find_denormalize_mapping(typed_jaxpr.jaxpr, constvar_state)
    denorm_map, denorm_valid_vars, _ = denorm_rec_state.state

    invar = typed_jaxpr.jaxpr.invars[0]
    outvar = typed_jaxpr.jaxpr.outvars[0]

    # Proper mapping of the output to the input.
    assert len(denorm_map) == 1
    assert outvar in denorm_map
    assert denorm_map[outvar][0] == case["expected_op"]
    assert denorm_map[outvar][1] == invar
    # Input is a valid denorm variable (which could be propagated in sub-jaxpr).
    assert denorm_valid_vars == {invar, outvar}


denorm_linear_op_propagating = [
    {"fn": lambda x: -(x + 1.0), "expected_op": jax_lax_identity},
    {"fn": lambda x: x + (x + 1.0), "expected_op": jax_lax_identity},
    {"fn": lambda x: x - (x + 1.0), "expected_op": jax_lax_identity},
    {"fn": lambda x: 2.0 * (x + 1.0), "expected_op": jax_lax_identity},
    {"fn": lambda x: (x + 1.0) * 2.0, "expected_op": jax_lax_identity},
    {"fn": lambda x: (x + 1.0) / 2.0, "expected_op": jax_lax_identity},
    {"fn": lambda x: jnp.expand_dims(1.0 - x, axis=0), "expected_op": jax.lax.neg},
    {"fn": lambda x: jnp.reshape(1.0 - x, (1, 1)), "expected_op": jax.lax.neg},
    {
        "fn": lambda x: jnp.squeeze(jnp.expand_dims(1.0 - x, axis=0)),
        "expected_op": jax.lax.neg,
    },
    {"fn": lambda x: jnp.full((2,), 2.0) * (1.0 - x), "expected_op": jax.lax.neg},
    {"fn": lambda x: (1.0 - x) / (jnp.ones((2,)) * 2.0), "expected_op": jax.lax.neg},
    # Typical case of support in distribution logpdf.
    {
        "fn": lambda x: jax.lax.select(1.0 > 0.0, 1.0 - x, -np.inf),
        "expected_op": jax.lax.neg,
    },
]


@pytest.mark.parametrize("case", denorm_linear_op_propagating)
def test__jaxpr_find_denormalize_mapping__linear_op_propagating__proper_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    constvars = {v: ConstVarInfo(False, True) for v in typed_jaxpr.jaxpr.constvars}
    constvar_state = jaxpr_find_constvars(typed_jaxpr.jaxpr, constvars)

    denorm_rec_state = jaxpr_find_denormalize_mapping(typed_jaxpr.jaxpr, constvar_state)
    denorm_map, denorm_valid_vars, constvar_rec_state = denorm_rec_state.state

    invar = typed_jaxpr.jaxpr.invars[0]
    # Proper mapping of the output to the input.
    assert len(denorm_map) == 1
    map_op, map_invar = list(denorm_map.values())[0]
    assert map_op == case["expected_op"]
    assert map_invar == invar
    # Input is a valid denorm variable (which could be propagated in sub-jaxpr).
    assert invar in denorm_valid_vars
    # Returned constvar sub states should be empty: consumed in the visitor loop.
    assert len(constvar_rec_state.sub) == 0


denorm_sub_jaxprs_propagating = [
    {"fn": lambda x: jax.jit(lambda y: 1.0 - y)(x), "expected_op": jax.lax.neg},
    # {"fn": lambda x: jnp.full((2,), 2.0) * (1.0 - x), "expected_op": jax.lax.neg},
    # {"fn": lambda x: (1.0 - x) / (jnp.ones((2,)) * 2.0), "expected_op": jax.lax.neg},
]


@pytest.mark.parametrize("case", denorm_sub_jaxprs_propagating)
def test__jaxpr_find_denormalize_mapping__sub_jaxprs_propagating__proper_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    constvars = {v: ConstVarInfo(False, True) for v in typed_jaxpr.jaxpr.constvars}
    constvar_state = jaxpr_find_constvars(typed_jaxpr.jaxpr, constvars)

    denorm_rec_state = jaxpr_find_denormalize_mapping(typed_jaxpr.jaxpr, constvar_state)
    denorm_map, denorm_valid_vars, constvar_rec_state = denorm_rec_state.state

    # Proper mapping of the output to the input.
    # assert len(denorm_map) == 1
    # map_op, map_invar = list(denorm_map.values())[0]
    # assert map_op == case["expected_op"]
    # assert map_invar == invar

    # Input is a valid denorm variable (which could be propagated in sub-jaxpr).
    invar = typed_jaxpr.jaxpr.invars[0]
    assert invar in denorm_valid_vars
    # Returned constvar sub states should be empty: consumed in the visitor loop.
    assert len(constvar_rec_state.sub) == 0


denorm_non_linear_fn = [
    {"fn": lambda x: jnp.sin(x + 1.0)},
    {"fn": lambda x: jnp.abs(x + 1.0)},
    {"fn": lambda x: jnp.exp(x + 1.0)},
    {"fn": lambda x: x + jnp.sin(x + 1.0)},
    {"fn": lambda x: x * (x + 1.0)},
]


@pytest.mark.parametrize("case", denorm_non_linear_fn)
def test__jaxpr_find_denormalize_mapping__non_linear_fn__empty_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    constvars = {v: ConstVarInfo(False, True) for v in typed_jaxpr.jaxpr.constvars}
    constvar_state = jaxpr_find_constvars(typed_jaxpr.jaxpr, constvars)

    invar = typed_jaxpr.jaxpr.invars[0]
    denorm_rec_state = jaxpr_find_denormalize_mapping(typed_jaxpr.jaxpr, constvar_state)
    denorm_map, denorm_valid_vars, _ = denorm_rec_state.state
    # Not simplifying mapping found.
    assert len(denorm_map) == 0
    # Denormalization not propagating to the input.
    assert invar not in denorm_valid_vars


denormalize_simple_test_cases = [
    {"fn": lambda x: x + 1.0, "denorm_fn": lambda x: x, "inval": 2.0},
    {
        "fn": lambda x: 2.0 - jnp.sin(x + 1.0),
        "denorm_fn": lambda x: -jnp.sin(x + 1.0),
        "inval": 2.0,
    },
    {
        "fn": lambda x: 2.0 + x - jnp.sin(x + 1.0),
        "denorm_fn": lambda x: x - jnp.sin(x + 1.0),
        "inval": 2.0,
    },
    {
        "fn": lambda x: np.sum(x + 2.0),
        "denorm_fn": lambda x: np.sum(x),
        "inval": np.ones((10,)),
    },
    {
        "fn": lambda x: (x + 1.0) * 2.0,
        "denorm_fn": lambda x: x * 2.0,
        "inval": 3.0,
    },
    {
        "fn": lambda x: (x + 1.0) / 2.0,
        "denorm_fn": lambda x: x / 2.0,
        "inval": 3.0,
    },
    {
        "fn": lambda x: jnp.squeeze(jnp.expand_dims(1.0 - x, axis=0)),
        "denorm_fn": lambda x: jnp.squeeze(jnp.expand_dims(-x, axis=0)),
        "inval": 3.0,
    },
    {
        "fn": lambda x: jax.lax.select(1.0 > 0.0, 1.0 - x, -np.inf),
        "denorm_fn": lambda x: jax.lax.select(1.0 > 0.0, -x, -np.inf),
        "inval": 2.0,
    },
    # {
    #     "fn": lambda x: jax.jit(lambda y: 1.0 - y)(x),
    #     "denorm_fn": lambda x: -x,
    #     "inval": jnp.array(2.0),
    # },
]


@pytest.mark.parametrize("case", denormalize_simple_test_cases)
def test__denormalize__simple_methods__proper_simplication(case):
    denorm_fn = denormalize(case["fn"])
    expected_denorm_fn = case["denorm_fn"]
    inval = case["inval"]

    denorm_fn_outval = denorm_fn(inval)
    assert np.allclose(denorm_fn_outval, expected_denorm_fn(inval))
