import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from pgen.config import Config
from pgen.net.mlp import MLP
from pgen.net.cnn import CNN


def get_network(cfg: Config, in_shape, out_shape, key):

    n_cfg = cfg.net
    out_features = out_shape
    if n_cfg.arch == "mlp":
        net = MLP(
            features=n_cfg.features,
            out_features=out_features,
            activation=n_cfg.activation,
            use_bias=n_cfg.use_bias,
            kernel_init=n_cfg.kernel_init,
            bias_init=n_cfg.bias_init,
            param_dtype=n_cfg.param_dtype,
            flatten=n_cfg.flatten,
            squeeze=n_cfg.squeeze,
        )

        in_dummy = jnp.ones(in_shape)

    elif n_cfg.arch == "cnn":
        dim = len(in_shape) - 2
        kernel_size = (n_cfg.kernel_size,) * dim

        net = CNN(
            features=n_cfg.features,
            out_features=out_features,
            activation=n_cfg.activation,
            kernel_size=kernel_size,
            padding=n_cfg.padding,
            use_bias=n_cfg.use_bias,
            kernel_init=n_cfg.kernel_init,
            bias_init=n_cfg.bias_init,
            param_dtype=n_cfg.param_dtype,
            pool=n_cfg.pool,
            squeeze=n_cfg.squeeze,
            norm_layer=n_cfg.norm_layer,
            dropout=n_cfg.dropout,
        )

        in_dummy = jnp.ones(in_shape)

    tabulate_fn = nn.tabulate(net, key)
    print(tabulate_fn(in_dummy))

    return net, in_dummy
