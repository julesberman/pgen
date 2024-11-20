import flax.linen as nn
import flax.linen
import jax
import jax.numpy as jnp
from einops import rearrange
from flax.linen import initializers
import flax


def get_activation(activation: str):
    if activation == "relu":
        a = jax.nn.relu
    elif activation == "tanh":
        a = jax.nn.tanh
    elif activation == "sigmoid":
        a = jax.nn.sigmoid
    elif activation == "elu":
        a = jax.nn.elu
    elif activation == "selu":
        a = jax.nn.selu
    elif activation == "gelu":
        a = jax.nn.gelu
    elif activation == "swish":
        a = jax.nn.swish
    elif activation == "sin":
        a = jnp.sin
    elif activation == "hswish":
        a = jax.nn.hard_swish
    else:
        raise KeyError(f"activation '{activation}' not found.")
    return a


def get_init(init: str) -> initializers.Initializer:

    if init is None or init == "lecun":
        w = initializers.lecun_normal()
    elif init == "ortho":
        w = initializers.orthogonal()
    elif init == "he":
        w = initializers.he_normal()
    elif init == "zero":
        w = initializers.zeros_init()
    elif init == "zero":
        w = initializers.zeros_init()
    else:
        raise KeyError(f"init '{init}' not found.")
    return w


def get_param_dtype(dtype: str) -> jnp.dtype:

    if "64" in dtype:
        return jnp.float64
    elif "32" in dtype:
        return jnp.float32
    elif "16" in dtype:
        return jnp.float16
    else:
        raise KeyError(f"param_dtype '{dtype}' not found.")


def get_norm_layer(norm_layer: str, param_dtype):
    if norm_layer == "layer":
        a = flax.linen.LayerNorm(param_dtype=param_dtype)
    elif norm_layer == "group":
        a = flax.linen.GroupNorm(param_dtype=param_dtype)
    elif norm_layer == "instance":
        a = flax.linen.InstanceNorm(param_dtype=param_dtype)
    else:
        raise KeyError(f"norm_layer '{norm_layer}' not found.")
    return a


class FixedDense(nn.Dense):
    def param(self, name, init_fn, *args, **kwargs):
        # Store parameters in the 'fixed_params' collection instead of 'params'
        return self.variable("fixed_params", name, init_fn, *args, **kwargs).value
