import hydra
import jax
import pgen.io.result as R
from pgen.config import Config
from pgen.data.get import get_dataset
from pgen.io.save import save_results
from pgen.io.setup import setup
from pgen.loss.get import get_loss_fns
from pgen.net.get import get_network
from pgen.train.test import test_model
from pgen.train.train import train_model

import time


@hydra.main(version_base=None, config_name="default")
def run(cfg: Config) -> None:

    key = setup(cfg)

    (train_ds, val_ds, test_ds), in_shape, out_shape = get_dataset(cfg)
    net, input_dummy = get_network(cfg, in_shape, out_shape, key)
    loss_fn, acc_fn = get_loss_fns(cfg, net)

    name = cfg.name
    for run_i in range(cfg.runs):
        t = time.time()
        print(f"===================== RUN {run_i:04} =====================")
        cfg.run_i = run_i
        cfg.name = f"{run_i}_{name}"
        R.RESULT["run_i"] = run_i
        key = jax.random.fold_in(key, run_i)
        params_init = net.init(key, input_dummy, train=False)

        last_params, opt_params = train_model(
            cfg, net, params_init, train_ds, val_ds, loss_fn, acc_fn, key
        )

        opt_m = test_model(cfg, opt_params, test_ds, loss_fn, acc_fn, prefix="opt")
        # last_m = test_model(cfg, last_params, test_ds, loss_fn, acc_fn, prefix="last")
        save_results(R.RESULT, cfg)
        print(f"=====================================================")
        elapsed_time = time.time() - t
        print(f"RUN {run_i:04} took {elapsed_time:.2f}\n")


if __name__ == "__main__":
    run()
