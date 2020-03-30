import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from mcx import sample_forward
from mcx.core import compile_to_logpdf


__all__ = ["sample", "generate"]


class sample(object):
    def __init__(
        self, rng_key, model, program, num_warmup_steps=1000, num_chains=4, **kwargs
    ):
        """ Initialize the sampling runtime.
        """
        self.program = program
        self.num_chains = num_chains
        self.rng_key = rng_key

        init, warmup, build_kernel, to_trace = self.program

        print("Initialize the sampler\n")

        validate_conditioning_variables(model, **kwargs)
        loglikelihood = build_loglikelihood(model, **kwargs)

        print("Find initial states...")
        initial_position, unravel_fn = get_initial_position(
            rng_key, model, num_chains, **kwargs
        )
        loglikelihood = flatten_loglikelihood(loglikelihood, unravel_fn)
        initial_state = jax.vmap(init, in_axes=(0, None))(
            initial_position, jax.value_and_grad(loglikelihood)
        )

        print("Warmup the chains...")
        parameters, state = warmup(initial_state, loglikelihood, num_warmup_steps)

        print("Compile the log-likelihood...")
        loglikelihood = jax.jit(loglikelihood)

        print("Build and compile the inference kernel...")
        kernel = build_kernel(loglikelihood, parameters)
        kernel = jax.jit(kernel)

        self.kernel = kernel
        self.state = state
        self.to_trace = to_trace
        self.unravel_fn = unravel_fn

    def run(self, num_samples=1000):
        _, self.rng_key = jax.random.split(self.rng_key)

        @jax.jit
        def update_chains(state, rng_key):
            keys = jax.random.split(rng_key, self.num_chains)
            new_states, info = jax.vmap(self.kernel, in_axes=(0, 0))(keys, state)
            return new_states

        state = self.state
        chain = []

        rng_keys = jax.random.split(self.rng_key, num_samples)
        with tqdm(rng_keys, unit="samples") as progress:
            progress.set_description(
                "Collecting {:,} samples across {:,} chains".format(
                    num_samples, self.num_chains
                ),
                refresh=False,
            )
            for key in progress:
                state = update_chains(state, key)
                chain.append(state)
        self.state = state

        trace = self.to_trace(chain, self.unravel_fn)

        return trace


def generate(rng_key, model, program, num_warmup_steps=1000, num_chains=4, **kwargs):
    """ The generator runtime """

    init, warmup, build_kernel, to_trace = program

    print("Initialize the sampler\n")

    validate_conditioning_variables(model, **kwargs)
    loglikelihood = build_loglikelihood(model, **kwargs)

    print("Find initial states...")
    initial_position, unravel_fn = get_initial_position(
        rng_key, model, num_chains, **kwargs
    )
    loglikelihood = flatten_loglikelihood(loglikelihood, unravel_fn)
    initial_state = jax.vmap(init, in_axes=(0, None))(
        initial_position, jax.value_and_grad(loglikelihood)
    )

    print("Warmup the chains...")
    parameters, state = warmup(initial_state, loglikelihood, num_warmup_steps)

    print("Compile the log-likelihood...")
    loglikelihood = jax.jit(loglikelihood)

    print("Build and compile the inference kernel...")
    kernel = build_kernel(loglikelihood, parameters)
    kernel = jax.jit(kernel)

    state = initial_state
    while True:
        _, rng_key = jax.random.split(rng_key)

        keys = jax.random.split(rng_key, num_chains)
        new_states = jax.vmap(kernel)(keys, state)

        yield new_states


def validate_conditioning_variables(model, **kwargs):
    """ Check that all variables passed as arguments to the sampler
    are random variables or arguments to the sampler. And converserly
    that all of the model definition's positional arguments are given
    a value.
    """
    conditioning_vars = set(kwargs.keys())
    model_randvars = set(model.random_variables)
    model_args = set(model.arguments)
    available_vars = model_randvars.union(model_args)

    # The variables passed as an argument to the initialization (variables
    # on which the logpdf is conditionned) must be either a random variable
    # or an argument to the model definition.
    if not available_vars.issuperset(conditioning_vars):
        unknown_vars = list(conditioning_vars.difference(available_vars))
        unknown_str = ", ".join(unknown_vars)
        raise AttributeError(
            "You passed a value for {} which are neither random variables nor arguments to the model definition.".format(
                unknown_str
            )
        )

    # The user must provide a value for all of the model definition's
    # positional arguments.
    model_posargs = set(model.posargs)
    if model_posargs.difference(conditioning_vars):
        missing_vars = model_posargs.difference(conditioning_vars)
        missing_str = ", ".join(missing_vars)
        raise AttributeError(
            "You need to specify a value for the following arguments: {}".format(
                missing_str
            )
        )


def build_loglikelihood(model, **kwargs):
    artifact = compile_to_logpdf(model.graph, model.namespace)
    logpdf = artifact.compiled_fn
    loglikelihood = jax.partial(logpdf, **kwargs)
    return loglikelihood


def get_initial_position(rng_key, model, num_chains, **kwargs):
    conditioning_vars = set(kwargs.keys())
    model_randvars = set(model.random_variables)
    to_sample_vars = model_randvars.difference(conditioning_vars)

    samples = sample_forward(rng_key, model, num_samples=num_chains, **kwargs)
    initial_positions = dict((var, samples[var]) for var in to_sample_vars)

    # A naive way to go about flattening the positions is to transform the
    # dictionary of arrays that contain the parameter value to a list of
    # dictionaries, one per position and then unravel the dictionaries.
    # However, this approach takes more time than getting the samples in the
    # first place.
    #
    # Luckily, JAX first sorts dictionaries by keys
    # (https://github.com/google/jax/blob/master/jaxlib/pytree.cc) when
    # raveling pytrees. We can thus ravel and stack parameter values in an
    # array, sorting by key; this gives our flattened positions. We then build
    # a single dictionary that contains the parameters value and use it to get
    # the unraveling function using `unravel_pytree`.
    positions = np.stack(
        [np.ravel(samples[s]) for s in sorted(initial_positions.keys())], axis=1
    )

    sample_position_dict = {
        parameter: values[0] for parameter, values in initial_positions.items()
    }
    _, unravel_fn = ravel_pytree(sample_position_dict)

    return positions, unravel_fn


def flatten_loglikelihood(logpdf, unravel_fn):
    def flattened_logpdf(array):
        kwargs = unravel_fn(array)
        return logpdf(**kwargs)

    return flattened_logpdf
