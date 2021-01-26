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
)


def test__jaxpr_find_constvars__propagate_constants():
    def foo(x):
        return x + np.ones((2,)) + np.exp(2.0)

    typed_jaxpr = jax.make_jaxpr(foo)(1.0)

    # All inputs consts, outputs should be consts!
    constvars = jaxpr_find_constvars(
        typed_jaxpr.jaxpr, typed_jaxpr.jaxpr.invars + typed_jaxpr.jaxpr.constvars
    )
    for outvar in typed_jaxpr.jaxpr.outvars:
        assert outvar in constvars


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
]


@pytest.mark.parametrize("case", denorm_linear_op_propagating)
def test__jaxpr_find_denormalize_mapping__linear_op_propagating__proper_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    denorm_map = jaxpr_find_denormalize_mapping(
        typed_jaxpr.jaxpr, typed_jaxpr.jaxpr.constvars
    )
    invar = typed_jaxpr.jaxpr.invars[0]

    # Proper mapping of the output to the input.
    assert len(denorm_map) == 1
    map_op, map_invar = list(denorm_map.values())[0]
    assert map_op == case["expected_op"]
    assert map_invar == invar


denorm_non_linear_fn = [
    {"fn": lambda x: np.sin(x + 1.0)},
    {"fn": lambda x: np.abs(x + 1.0)},
    {"fn": lambda x: np.exp(x + 1.0)},
    {"fn": lambda x: x * (x + 1.0)},
]


@pytest.mark.parametrize("case", denorm_non_linear_fn)
def test__jaxpr_find_denormalize_mapping__non_linear_fn__empty_mapping(case):
    typed_jaxpr = jax.make_jaxpr(case["fn"])(1.0)
    denorm_map = jaxpr_find_denormalize_mapping(
        typed_jaxpr.jaxpr, typed_jaxpr.jaxpr.constvars
    )
    assert len(denorm_map) == 0


denormalize_test_cases = [
    {"fn": lambda x: x + 1.0, "exp_denorm_fn": lambda x: x},
    {
        "fn": lambda x: 2.0 - np.sin(x + 1.0),
        "exp_denorm_fn": lambda x: -np.sin(x + 1.0),
    },
]


@pytest.mark.parametrize("case", denormalize_test_cases)
def test__denormalize__proper_simplication(case):
    denorm_fn = denormalize(case["fn"])
    exp_denorm_fn = case["exp_denorm_fn"]

    inval = 1.0
    assert np.allclose(denorm_fn(inval), exp_denorm_fn(inval))
