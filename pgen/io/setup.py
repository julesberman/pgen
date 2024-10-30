
import os
import random
import secrets

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.lib.xla_bridge import get_backend
from omegaconf import OmegaConf

import pgen.io.result as R
from pgen.config import Config
from pgen.io.result import init_result
from pgen.utils.misc import unique_id


def setup(config: Config):

    if config.info is not None:
        print(f'INFO: {config.info}')

    # init global results obj
    init_result()

    # log.info config
    print('\nCONFIG')
    print(OmegaConf.to_yaml(config))

    print(f'name: {config.name}')

    uuid = unique_id(10)
    print(f'uuid: {uuid}')
    R.RESULT['uuid'] = uuid

    if config.x64:
        jax.config.update("jax_enable_x64", True)
        print('enabling x64')
    else:
        jax.config.update("jax_enable_x64", False)

    if config.platform is not None:
        jax.config.update('jax_platform_name', config.platform)

    if config.debug_nans:
        jax.config.update("jax_debug_nans", True)

    # # if oop error see: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    print(
        f'platform: {get_backend().platform} — device_count: {jax.local_device_count()}')

    # list of available devices (CPUs, GPUs, TPUs)
    devices = jax.devices()
    for gpu in devices:
        print(f"host_{gpu.id}: {gpu.device_kind}")
        R.RESULT[f'host_{gpu.id}'] = gpu.device_kind

    if config.advanced_flags:
        os.environ.update({
            "NCCL_LL128_BUFFSIZE": "-2",
            "NCCL_LL_BUFFSIZE": "-2",
            "NCCL_PROTO": "SIMPLE,LL,LL128",
        })

    # prep tfds
    # ensure TF does not see GPU and grab all GPU memory.
    tf.config.set_visible_devices([], device_type='GPU')

   # set random seed, if none use random random seed
    if config.seed == -1:
        config.seed = secrets.randbelow(1e5)
        print(f'seed: {config.seed}')
    seed = config.seed
    key = jax.random.PRNGKey(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    return key
