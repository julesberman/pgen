import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import pgen.io.result as R
from pgen.config import Config


def get_score_loss(net, norm_fn, noise_fn, noise_conditional=True):

    def denoiser_loss(params, x, sigma, key):
        noise = noise_fn(key, sigma)
        if noise_conditional:
            v = net.apply(params, x, sigma)
        else:
            v = net.apply(params, x)

        x_tilde = x + noise
        direction = v + (x_tilde - x) / sigma**2
        l = sigma**2 * 0.5 * norm_fn(direction)

        return l

    return denoiser_loss


def get_sigmas(L, start=1, end=1e-2):
    sigmas = jnp.asarray(
        [start * (end / start) ** (i / (L - 1)) for i in range(L)]
    ).reshape(-1, 1)
    return sigmas
