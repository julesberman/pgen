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


def test_model(cfg: Config, params, test_ds, loss_fn, acc_fn):

    # init optimizer and training state
    n_test_batches = len(test_ds)

    def fmt_dict(d):
        return {k: f"{v:.3f}" for (k, v) in d.items()}

    accumlator = Accumlator()
    pbar = tqdm(total=n_test_batches, dynamic_ncols=True, colour="magenta")
    pbar.set_description(f"[test]")
    for images, labels in test_ds:
        test_loss, logits = loss_fn(params, images, labels)
        test_acc = acc_fn(logits, labels)

        stats = {"test_loss": test_loss.item(), "test_acc": test_acc.item()}
        accumlator.add(stats)
        pbar.set_postfix(fmt_dict(stats))
        pbar.update(1)
    accumlator.save()
    pbar.close()

    metrics = accumlator.streams

    for k, v in metrics.items():
        v = v[0]
        print(f"{k}: {v:.3f}", end=" | ")
        R.RESULT[f"{k}"] = v

    print("\n")
    return metrics
