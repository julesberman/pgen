import hydra

import pgen.io.result as R
from pgen.config import Config
from pgen.data.get import get_dataset
from pgen.io.save import save_results
from pgen.io.setup import setup
from pgen.loss.get import get_loss_fns
from pgen.net.get import get_network
from pgen.train.test import test_model
from pgen.train.train import train_model


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key = setup(cfg)

    (train_ds, val_ds, test_ds), in_shape, out_shape = get_dataset(cfg)
    net, params_init = get_network(cfg, in_shape, out_shape, key)
    loss_fn, acc_fn = get_loss_fns(cfg, net)

    last_params, opt_params = train_model(
        cfg, net, params_init, train_ds, val_ds, loss_fn, acc_fn)

    opt_m = test_model(cfg, opt_params, test_ds, loss_fn, acc_fn, prefix='opt')
    last_m = test_model(cfg, last_params, test_ds,
                        loss_fn, acc_fn, prefix='last')
    save_results(R.RESULT, cfg)


if __name__ == "__main__":
    run()
