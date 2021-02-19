from jax import lax
from jax import numpy as jnp


def promote_shapes(*args):
    """Preprend implicit leading singleton dimensions.

    This is necessary for proper numpy broadcasting. Consider the following
    model:

        >>> def toy_model():
        ...     sigma = jnp.array([1, 2, 3])
        ...     x <~ Normal(0, sigma) # shape (n_samples, 3)
        ...     q <~ Normal(1, 1) # shape (n_samples,)
        ...     y <~ Normal(x, q)

    When sampling, the last line will trigger a broadcasting error since
    Numpy's default is to broadcast (n,) to (1,n). To avoid this we prepend
    (1,) to the shape of `q`.

    We add as many leading singleton dimensions as necessary for all variables
    to have the same number of dimensions. See
    `jax.numpy.lax_numpy._promote_shapes` for the reference implementation.

    """
    if len(args) < 2:
        return args
    else:
        shapes = [jnp.shape(arg) for arg in args]
        batch_shape = lax.broadcast_shapes(*shapes)
        num_dims = len(batch_shape)
        return (
            batch_shape,
            [
                jnp.reshape(arg, (1,) * (num_dims - len(s)) + s)
                if len(s) < num_dims
                else arg
                for arg, s in zip(args, shapes)
            ],
        )
