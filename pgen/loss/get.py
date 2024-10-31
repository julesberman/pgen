import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import pgen.io.result as R
from pgen.config import Config


def get_loss_fns(cfg: Config, net):

    if cfg.loss.loss == "bce":

        def loss_fn(params, images, labels, train=False, key=None):
            rngs = {"dropout": key}
            if key is None:
                rngs = None
            logits = net.apply(params, images, train=train, rngs=rngs)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=labels
            ).mean()
            return loss, logits

    if cfg.loss.acc == "classify":

        def acc_fn(logits, labels):
            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(preds == labels)
            return accuracy

    return loss_fn, acc_fn
