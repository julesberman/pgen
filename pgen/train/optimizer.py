from functools import partial

import optax

from pgen.config import Config

str_to_opt = {
    'adam': optax.adam,
    'adamw': optax.adamw,
    'adamww': partial(optax.adamw, weight_decay=1e-3),
    'sgd':  optax.sgd,
    'lion': optax.lion,
}


def get_optimizer(cfg: Config, steps):

    o_cfg = cfg.train
    # adds warm up cosine decay
    if cfg.train.scheduler is not None:
        learning_rate = optax.cosine_decay_schedule(
            init_value=o_cfg.lr,
            decay_steps=steps,
            alpha=0.0
        )
    else:
        learning_rate = o_cfg.lr

    opti_f = str_to_opt[o_cfg.optimizer]
    train = opti_f(learning_rate=learning_rate)

    return train
