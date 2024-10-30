import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm

import pgen.io.result as R
from pgen.config import Config
from pgen.train.optimizer import get_optimizer
from pgen.train.utils import Accumlator


def train_model(cfg: Config, net, params, train_ds, val_ds, loss_fn, acc_fn):

    # init optimizer and training state
    num_epochs = cfg.train.epochs
    n_train_batches = len(train_ds)
    n_val_batches = len(val_ds)
    total_steps = n_train_batches * num_epochs
    optimizer = get_optimizer(cfg, total_steps)

    state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=optimizer)

    @jax.jit
    def train_step(state, images, labels):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params, images, labels)
        state = state.apply_gradients(grads=grads)
        acc = acc_fn(logits, labels)
        return state, loss, acc

    opt_params = params
    opt_acc = 0.0
    accumlator = Accumlator()

    def fmt_dict(d): return {k: f'{v:.3f}' for (k, v) in d.items()}
    # training loop
    pbar_t = tqdm(total=n_train_batches, dynamic_ncols=True, colour='blue')
    pbar_v = tqdm(total=n_val_batches, dynamic_ncols=True, colour='green')
    for epoch in range(1, num_epochs + 1):

        pbar_t.reset(total=n_train_batches)
        pbar_t.set_description(
            f"[train] epoch {epoch}/{num_epochs}")
        for images, labels in train_ds:
            state, train_loss, train_acc = train_step(state, images, labels)
            train_stats = {'train_loss': train_loss.item(),
                           'train_acc': train_acc.item()}
            accumlator.add(train_stats)
            pbar_t.set_postfix(fmt_dict(train_stats))
            pbar_t.update(1)
        accumlator.save()

        # validation phase
        pbar_v.reset(total=n_val_batches)
        pbar_v.set_description(
            f"[valid] epoch {epoch}/{num_epochs}")
        for images, labels in val_ds:

            val_loss, logits = loss_fn(state.params, images, labels)
            val_acc = acc_fn(logits, labels)

            # opt_val stopping
            if val_acc > opt_acc:
                opt_acc = val_acc
                opt_params = state.params

            val_stats = {'valid_loss': val_loss.item(),
                         'valid_acc': val_acc.item()}
            accumlator.add(val_stats)
            pbar_v.set_postfix(fmt_dict(val_stats))
            pbar_v.update(1)
        accumlator.save()

    pbar_t.close()
    pbar_v.close()

    metrics = accumlator.streams
    last_params = state.params

    # save results
    R.RESULT['last_params'] = last_params
    R.RESULT['opt_params'] = opt_params
    R.RESULT.update(metrics)

    print()
    return last_params, opt_params
