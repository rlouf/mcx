from jax import lax


def broadcast_batch_shape(*shapes):
    """Compute the batch shape by broadcasting the arguments.

    We use `lax.broadcast_shapes` to get the shape of broadcasted arguments. We
    default the batch shape to (1,) when the distribution is initiated with
    scalar values.

    To see why we need to do that, consider the following model:

        >>> def toy_model():
        ...     sigma = jnp.array([1, 2, 3])
        ...     x <~ Normal(0, sigma) # shape (n_samples, 3)
        ...     q <~ Normal(1, 1) # shape (n_samples,)
        ...     y <~ Normal(x, q)

    When sampling, the last line will trigger a broadcasting error since
    Numpy's default is to broadcast (n,) to (1,n). To avoid this we explicit
    the fact that a distribution initiated with scalar arguments has a batch
    size of 1.
    """
    broadcasted_shape = lax.broadcast_shapes(*shapes)
    if len(broadcasted_shape) == 0:
        return (1,)
    return broadcasted_shape
