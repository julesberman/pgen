
from dataclasses import dataclass, field
from typing import Any, List, Union

from hydra.core.config_store import ConfigStore

from pgen.utils.misc import epoch_time, unique_id

SWEEP = {}
SLURM_CONFIG = {
    'timeout_min': 60*2,
    'cpus_per_task': 4,
    'mem_gb': 200,
    # 'gpus_per_node': 1,
    'gres': 'gpu',
    # 'account': 'extremedata'
}


@dataclass
class Network:
    arch: str = 'mlp'
    features: List[int] = field(default_factory=lambda: [35]*5)
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    param_dtype: str = 'float32'
    flatten: bool = True


@dataclass
class Train:
    lr: float = 2e-3
    epochs: int = 25
    scheduler: bool = True
    optimizer: str = 'adamw'
    save_params_history: bool = False


@dataclass
class Dataset:
    name: str = 'mnist'
    batch_size: int = 256
    val_split: float = 0.15
    data_dir: str | None = None
    cache: bool = True


@dataclass
class Loss:
    acc: str = 'classify'
    loss: str = 'bce'


@dataclass
class Config:

    net: Network = field(default_factory=Network)

    train: Train = field(default_factory=Train)
    dataset: Dataset = field(default_factory=Dataset)
    loss: Loss = field(default_factory=Loss)

    name: str = field(
        default_factory=lambda: f'{unique_id(4)}_{epoch_time(2)}')
    x64: bool = False  # whether to use 64 bit precision in jax

    platform: Union[str, None] = None  # gpu or cpu, None will let jax default
    # output_dir: str = './results/${hydra.job.name}'  # where to save results, if None nothing is saved

    seed: int = 1
    debug_nans: bool = False  # whether to debug nans
    # set advanced flags https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#nccl-flags
    advanced_flags: bool = False
    # optional info about details of the experiment
    info: Union[str, None] = None

    # hydra config configuration
    hydra: Any = field(default_factory=lambda: hydra_config)
    defaults: List[Any] = field(default_factory=lambda: defaults)


##########################
## hydra settings stuff ##
##########################
defaults = [
    # https://hydra.cc/docs/tutorials/structured_config/defaults/
    # "_self_",
    {"override hydra/launcher": "submitit_slurm"},
]


hydra_config = {
    # sets the out dir from config.problem and id
    "run": {
        "dir": "results/${dataset.name}/single/${name}"
    },
    "sweep": {
        "dir": "results/${dataset.name}/multi/${name}"
    },

    # "mode": get_mode(),
    "sweeper": {
        "params": {
            **SWEEP
        }
    },
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {
        **SLURM_CONFIG
    },
    "job": {
        "env_set": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
    }
}


##################################
## problem wise default configs ##
##################################


cs = ConfigStore.instance()
cs.store(name="default", node=Config)

mnist_config = Config(dataset=Dataset(name='mnist'),
                      train=Train(epochs=25))
cifar10_config = Config(dataset=Dataset(name='cifar10'),
                        train=Train(epochs=100))


cs.store(name="mnist", node=mnist_config)
cs.store(name="cifar10", node=cifar10_config)
