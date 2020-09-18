import pytest

from mcx.inference.warmup import warmup_schedule


@pytest.mark.parametrize(
    "num_steps, expected_schedule",
    [
        (10, [(0, 9)]),  # num_steps < 20
        (100, [(0, 14), (15, 89), (90, 99)]),  # num_steps < sum(window sizes)
        (
            1000,
            [
                (0, 74),
                (75, 99),
                (100, 149),
                (150, 249),
                (250, 449),
                (450, 949),
                (950, 999),
            ],
        ),
    ],
)
def test_hmc_warmup_schedule(num_steps, expected_schedule):
    """Make sure that the scheduler behaves as expected for
    different number of warmup steps.
    """
    schedule = warmup_schedule(num_steps)
    assert schedule == expected_schedule
