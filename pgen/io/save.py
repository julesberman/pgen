import pickle
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from jax.experimental.host_callback import id_print, id_tap

import pgen.io.result as R
from pgen.config import Config
from pgen.io.utils import (convert_jax_to_numpy, convert_list_to_numpy,
                           flatten_config, save_pickle)


def consolidate_results(results: dict, cfg: Config):

    r_dict = convert_list_to_numpy(results)
    r_dict = convert_jax_to_numpy(r_dict)

    # flatten config
    args_d = flatten_config(cfg, '.')
    all_data = {**args_d, **r_dict}

    return all_data


def save_results(results: dict, cfg: Config):

    output_dir = HydraConfig.get().runtime.output_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_name = 'result'
    output_path = (output_dir / output_name).with_suffix(".pkl")

    data = consolidate_results(results, cfg)

    save_pickle(output_path, data)


def jit_save(data, key: str):

    def save_on_host(data, transforms):
        if not key in R.RESULT:
            R.RESULT[key] = []

        R.RESULT[key].append(data)

    id_tap(save_on_host, data)
