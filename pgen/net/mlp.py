from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from pgen.net.parts import get_activation, get_init, get_param_dtype


class MLP(nn.Module):
    features: List[int]
    out_features: int
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    param_dtype: str = 'float32'
    flatten: bool = True

    @nn.compact
    def __call__(self, x):
        kernel_init = get_init(self.kernel_init)
        bias_init = get_init(self.bias_init)
        param_dtype = get_param_dtype(self.param_dtype)
        A = get_activation(self.activation)

        if self.flatten:
            x = rearrange(x, 'b ... -> b (...)')

        for i, feats in enumerate(self.features):
            D = nn.Dense(feats, use_bias=self.use_bias, bias_init=bias_init,
                         kernel_init=kernel_init, param_dtype=param_dtype)
            x = D(x)
            x = A(x)

        D = nn.Dense(self.out_features, use_bias=self.use_bias, bias_init=bias_init,
                     kernel_init=kernel_init, param_dtype=param_dtype)
        x = D(x)

        return x
