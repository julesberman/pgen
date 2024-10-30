

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

import pgen.io.result as R
from pgen.config import Config


def get_loss_fns(cfg: Config, net):

    if cfg.loss.loss == 'bce':
        def loss_fn(params, images, labels):
            logits = net.apply(params, images)
            loss = optax.sigmoid_binary_cross_entropy(
                logits, labels).mean()
            return loss, logits

    if cfg.loss.acc == 'classify':
        def acc_fn(output, labels):
            labels = jnp.argmax(labels, axis=-1)
            logits = nn.sigmoid(output)
            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(preds == labels)
            return accuracy

    return loss_fn, acc_fn
