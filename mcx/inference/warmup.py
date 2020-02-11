"""Warming up the chain.

.. note:
    This is a "flat zone": all positions are 1D array.
"""


def hmc_warmup():
    def init():
        pass

    def update():
        pass


def warmup_schedule(num_steps, initial_buffer=75, first_window=25, final_buffer=50):
    """Returns an adaptation warmup schedule.

    The schedule below is intended to be as close as possible to Stan's _[1]. 
    The warmup period is split into three stages:

    1. An initial fast interval to reach the typical set.
    2. "Slow" parameters that require global information (typically covariance)
       are estimated in a series of expanding windows with no memory.
    3. Fast parameters are learned after the adaptation of the slow ones.

    See _[1] for a more detailed explanation.

    Parameters
    ----------
    num_warmup: int
        The number of warmup steps to perform.
    initial_buffer: int
        The width of the initial fast adaptation interval.
    first_window: int
        The width of the first slow adaptation interval. There are 5 such
        intervals; the width of a window interval is twice the size of the
        preceding.
    final_buffer: int
        The width of the final fast adaptation interval.

    References
    ----------
    .. [1]: Stan Reference Manual v2.22
            Section 15.2 "HMC Algorithm"
    """
    schedule = []

    # Handle the situations where the numbrer of warmup steps is smaller than
    # the sum of the buffers' widths
    if num_steps < 20:
        schedule.append((0, num_steps - 1))
        return schedule

    if initial_buffer + first_window + final_buffer > num_steps:
        initial_buffer = int(0.15 * num_steps)
        final_buffer = int(0.1 * num_steps)
        first_window = num_steps - initial_buffer - final_buffer

    # First stage: adaptation of fast parameters
    schedule.append((0, initial_buffer - 1))

    # Second stage: adaptation of slow parameters
    final_buffer_start = num_steps - final_buffer

    next_size = first_window
    next_start = initial_buffer
    while next_start < final_buffer_start:
        start, size = next_size, next_start
        if 3 * size <= final_buffer_start - start:
            next_size = 2 * size
        else:
            size = final_buffer_start - start
        next_start = start + size
        schedule.append((start, next_start - 1))

    # Last stage: adaptation of fast parameters
    schedule.append((final_buffer_start, num_steps - 1))

    return schedule
