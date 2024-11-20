from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from pgen.net.parts import get_activation, get_init, get_param_dtype, get_norm_layer


class MLP(nn.Module):
    features: List[int]
    out_features: int
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    param_dtype: str = "float32"
    flatten: bool = True
    squeeze: bool = True
    activate_last: bool = False
    norm_layer: str | None = None
    conditional: bool = False
    residual: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        kernel_init = get_init(self.kernel_init)
        bias_init = get_init(self.bias_init)
        param_dtype = get_param_dtype(self.param_dtype)
        A = get_activation(self.activation)

        last_x = None
        if self.conditional:
            conditional_features = 16
            x, cond_x = x

        if self.flatten:
            x = rearrange(x, "b ... -> b (...)")

        for i, feats in enumerate(self.features):
            D = nn.Dense(
                feats,
                use_bias=self.use_bias,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=param_dtype,
            )
            last_x = x
            x = D(x)
            x = A(x)
            if self.norm_layer is not None:
                N = get_norm_layer(self.norm_layer, param_dtype)
                x = N(x)

            if self.conditional:
                Hyper_Net_Head = MLP(
                    features=[conditional_features] * 2,
                    activate_last=False,
                    out_features=feats * 2,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    param_dtype=self.param_dtype,
                )
                s_b = Hyper_Net_Head(cond_x).reshape(-1, 2, feats)
                scale, bias = s_b[:, 0], s_b[:, 1]
                scale = scale.reshape(-1, feats)
                bias = bias.reshape(-1, feats)
                x = x * scale + bias

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        D = nn.Dense(
            self.out_features,
            use_bias=self.use_bias,
            bias_init=bias_init,
            kernel_init=kernel_init,
            param_dtype=param_dtype,
        )
        x = D(x)

        if self.activate_last:
            x = A(x)

        if self.squeeze:
            x = jnp.squeeze(x)
        return x


class Fixed_MLP(nn.Module):
    features: List[int]
    out_features: int
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    param_dtype: str = "float32"
    flatten: bool = True
    squeeze: bool = True
    fixed_seed: int = 1

    @nn.compact
    def __call__(self, x, train=False):
        kernel_init = get_init(self.kernel_init)
        bias_init = get_init(self.bias_init)
        param_dtype = get_param_dtype(self.param_dtype)
        A = get_activation(self.activation)

        if self.flatten:
            x = rearrange(x, "b ... -> b (...)")

        in_features = x.shape[-1]
        hidden_feats = self.features[0]
        key = jax.random.PRNGKey(self.fixed_seed)
        in_key, out_key = jax.random.split(key)

        # apply in random projection
        R_in = jax.random.normal(
            in_key, shape=(in_features, hidden_feats), dtype=param_dtype
        )
        R_in = R_in * jnp.sqrt(1 / in_features)
        x = jax.lax.dot_general(x, R_in, (((x.ndim - 1,), (0,)), ((), ())))
        x = A(x)

        for i, feats in enumerate(self.features):
            D = nn.Dense(
                feats,
                use_bias=self.use_bias,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=param_dtype,
            )
            x = D(x)
            x = A(x)

        # apply out random projection
        R_out = jax.random.normal(
            out_key, shape=(hidden_feats, self.out_features), dtype=param_dtype
        )
        R_out = R_out * jnp.sqrt(1 / hidden_feats)

        x = jax.lax.dot_general(x, R_out, (((x.ndim - 1,), (0,)), ((), ())))

        if self.squeeze:
            x = jnp.squeeze(x)
        return x
