from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from pgen.net.parts import get_activation, get_init, get_param_dtype, get_norm_layer
from pgen.net.mlp import MLP


class CNN(nn.Module):
    features: List[int]
    out_features: int
    activation: str = "swish"
    kernel_size: int | List[int] = 2
    padding: str = "SAME"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    param_dtype: str = "float32"
    pool: bool = True
    squeeze: bool = True
    norm_layer: str | None = None
    dropout: float = 0.0
    residual: bool = False
    conditional: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        kernel_init = get_init(self.kernel_init)
        bias_init = get_init(self.bias_init)
        param_dtype = get_param_dtype(self.param_dtype)
        A = get_activation(self.activation)

        last_x = None
        depth = len(self.features)

        if self.conditional:
            conditional_features = 16
            x, cond_x = x

            # Hyper_Net = MLP(
            #     features=[conditional_features] * 2,
            #     activate_last=True,
            #     out_features=35,
            #     activation=self.activation,
            #     use_bias=self.use_bias,
            #     kernel_init=self.kernel_init,
            #     bias_init=self.bias_init,
            #     param_dtype=self.param_dtype,
            # )
            # cond_x = Hyper_Net(cond_x)

        for i, feats in enumerate(self.features):
            C = nn.Conv(
                feats,
                kernel_size=self.kernel_size,
                padding=self.padding,
                use_bias=self.use_bias,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=param_dtype,
            )
            last_x = x

            x = C(x)
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
                scale = scale.reshape(-1, 1, 1, feats)
                bias = bias.reshape(-1, 1, 1, feats)
                # jax.debug.print('{}', scale)
                x = x * scale + bias

            if self.pool:
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

            if self.dropout > 0.0:
                D = nn.Dropout(self.dropout, deterministic=not train)
                x = D(x)

            if self.residual and last_x is not None:
                if x.shape == last_x.shape:
                    x = x + last_x

        if self.out_features > 0:

            x = x.reshape((x.shape[0], -1))
            D = nn.Dense(
                self.out_features,
                use_bias=self.use_bias,
                bias_init=bias_init,
                kernel_init=kernel_init,
                param_dtype=param_dtype,
            )
            x = D(x)

        if self.squeeze:
            x = jnp.squeeze(x)

        return x
