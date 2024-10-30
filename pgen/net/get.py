import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax.experimental.host_callback import id_print

from pgen.config import Config, Network
from pgen.net.mlp import MLP


def get_network(cfg: Config, in_shape, out_shape, key):

    n_cfg = cfg.net
    out_features = out_shape[1]
    if n_cfg.arch == 'mlp':
        net = MLP(features=n_cfg.features,
                  out_features=out_features,
                  activation=n_cfg.activation,
                  use_bias=n_cfg.use_bias,
                  kernel_init=n_cfg.kernel_init,
                  bias_init=n_cfg.bias_init,
                  param_dtype=n_cfg.param_dtype,
                  flatten=n_cfg.flatten)

        in_dummy = jnp.ones(in_shape)
        tabulate_fn = nn.tabulate(net, key)
        print(tabulate_fn(in_dummy))

    params_init = net.init(key, in_dummy)

    return net, params_init
