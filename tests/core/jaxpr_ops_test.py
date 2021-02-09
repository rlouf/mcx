from typing import TypedDict
import jax
import jax.lax
import numpy as onp
import pytest
from jax import numpy as np

from mcx.core.jaxpr_ops import (
    jax_lax_identity,
    jaxpr_find_constvars,
    jaxpr_find_denormalize_mapping,
    denormalize,
    ConstVarStatus,
)


find_constvars_test_functions = [
    # Simple constant propagation.
    {"fn": lambda x: x + np.ones((2,)) + np.exp(2.0), "status": ConstVarStatus.Unknown},
    # Handle properly jax.jit sub-jaxpr.
    {
        "fn": lambda x: jax.jit(lambda y: y + np.ones((2,)))(x) + np.exp(2.0),
        "status": ConstVarStatus.Unknown,
    },
    # Simple inf constant propagation.
    {"fn": lambda x: x + np.ones((2,)) + np.inf, "status": ConstVarStatus.NonFinite},
    # Handle properly jax.jit sub-jaxpr + inf constant.
    {
        "fn": lambda x: jax.jit(lambda y: y + np.full((2,), np.inf))(x) + np.exp(2.0),
        "status": ConstVarStatus.NonFinite,
    },
    # TODO: test pmap, while, scan, cond.
]


@pytest.mark.parametrize("case", find_constvars_test_functions)
def test__jaxpr_find_constvars__propagate_constants(case):
    test_fn = case["fn"]
    expected_status = case["status"]
    typed_jaxpr = jax.make_jaxpr(test_fn)(1.0)

    # All inputs consts, outputs should be consts!
    constvars = {v: ConstVarStatus.Unknown for v in typed_jaxpr.jaxpr.invars}
    constvars.update({v: ConstVarStatus.Unknown for v in typed_jaxpr.jaxpr.constvars})

    constvars = jaxpr_find_constvars(typed_jaxpr.jaxpr, constvars)
    for outvar in typed_jaxpr.jaxpr.outvars:
        assert outvar in constvars
        assert constvars[outvar] == expected_status


denorm_expected_add_mapping_op = [
    {"fn": lambda x: x + 1.0, "expected_op": jax_lax_identity},
    {"fn": lambda x: 1.0 + x, "expected_op": jax_lax_identity},
    {"fn": lambda x: x - 1.0, "expected_op": jax_lax_identity},
    {"fn": lambda x: 1.0 - x, "expected_op": jax.lax.neg},
]


@pytest.mark.parametrize("case", denorm_expected_add_mapping_op)
def test__jaxpr_find_denormalize_mapping__add_sub__proper_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    denorm_map = jaxpr_find_denormalize_mapping(
        typed_jaxpr.jaxpr, typed_jaxpr.jaxpr.constvars
    )
    invar = typed_jaxpr.jaxpr.invars[0]
    outvar = typed_jaxpr.jaxpr.outvars[0]

    # Proper mapping of the output to the input.
    assert len(denorm_map) == 1
    assert outvar in denorm_map
    assert denorm_map[outvar][0] == case["expected_op"]
    assert denorm_map[outvar][1] == invar


denorm_linear_op_propagating = [
    {"fn": lambda x: -(x + 1.0), "expected_op": jax_lax_identity},
    {"fn": lambda x: np.expand_dims(1.0 - x, axis=0), "expected_op": jax.lax.neg},
    {"fn": lambda x: np.reshape(1.0 - x, (1, 1)), "expected_op": jax.lax.neg},
    {
        "fn": lambda x: np.squeeze(np.expand_dims(1.0 - x, axis=0)),
        "expected_op": jax.lax.neg,
    },
    {"fn": lambda x: jax.jit(lambda y: 1.0 - y)(x), "expected_op": jax.lax.neg},
    {"fn": lambda x: np.full((2,), 2.0) * (1.0 - x), "expected_op": jax.lax.neg},
    {"fn": lambda x: (1.0 - x) / 2.0, "expected_op": jax.lax.neg},
]


@pytest.mark.parametrize("case", denorm_linear_op_propagating)
def test__jaxpr_find_denormalize_mapping__linear_op_propagating__proper_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    denorm_map = jaxpr_find_denormalize_mapping(
        typed_jaxpr.jaxpr, typed_jaxpr.jaxpr.constvars
    )
    invar = typed_jaxpr.jaxpr.invars[0]

    print(typed_jaxpr)

    # Proper mapping of the output to the input.
    assert len(denorm_map) == 1
    map_op, map_invar = list(denorm_map.values())[0]
    assert map_op == case["expected_op"]
    assert map_invar == invar


# denorm_non_linear_fn = [
#     {"fn": lambda x: np.sin(x + 1.0)},
#     {"fn": lambda x: np.abs(x + 1.0)},
#     {"fn": lambda x: np.exp(x + 1.0)},
#     {"fn": lambda x: x * (x + 1.0)},
# ]


# @pytest.mark.parametrize("case", denorm_non_linear_fn)
# def test__jaxpr_find_denormalize_mapping__non_linear_fn__empty_mapping(case):
#     typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
#     denorm_map = jaxpr_find_denormalize_mapping(
#         typed_jaxpr.jaxpr, typed_jaxpr.jaxpr.constvars
#     )
#     assert len(denorm_map) == 0


# denormalize_test_cases = [
#     {"fn": lambda x: x + 1.0, "denorm_fn": lambda x: x, "inval": 2.0},
#     {
#         "fn": lambda x: 2.0 - np.sin(x + 1.0),
#         "denorm_fn": lambda x: -np.sin(x + 1.0),
#         "inval": 2.0,
#     },
#     {
#         "fn": lambda x: 2.0 - np.sin(x + 1.0),
#         "denorm_fn": lambda x: -np.sin(x + 1.0),
#         "inval": 2.0,
#     },
#     {
#         "fn": lambda x: np.sum(x + 2.0),
#         "denorm_fn": lambda x: np.sum(x),
#         "inval": np.ones((10,)),
#     },
# ]


# @pytest.mark.parametrize("case", denormalize_test_cases)
# def test__denormalize__proper_simplication(case):
#     denorm_fn = denormalize(case["fn"])
#     expected_denorm_fn = case["denorm_fn"]
#     inval = case["inval"]
#     assert np.allclose(denorm_fn(inval), expected_denorm_fn(inval))
