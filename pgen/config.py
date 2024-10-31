from dataclasses import dataclass, field
from typing import Any, List, Union

from hydra.core.config_store import ConfigStore

from pgen.utils.misc import epoch_time, unique_id

SWEEP = {
    "runs": "128",
    "net.dropout": "0.0, 0.1, 0.2, 0.3, 0.4, 0.5",
    "train.lr": "2e-4, 5e-4, 1e-3, 2e-3, 5e-3",
}
SLURM_CONFIG = {
    "timeout_min": 60 * 4,
    "cpus_per_task": 4,
    "mem_gb": 100,
    "gpus_per_node": 1,
    "gres": "gpu",
    "account": "extremedata",
}


@dataclass
class Network:
    arch: str = "mlp"
    features: List[int] = field(default_factory=lambda: [64] * 5)
    kernel_size: int = 3
    activation: str = "swish"
    use_bias: bool = True
    kernel_init: str = "lecun"
    bias_init: str = "zero"
    param_dtype: str = "float32"
    flatten: bool = True
    padding: str = "SAME"
    norm_layer: str | None = None
    pool: bool = True
    squeeze: bool = True
    dropout: float = 0.2


@dataclass
class Train:
    lr: float = 2e-3
    epochs: int = 25
    scheduler: bool = True
    optimizer: str = "adamw"
    save_params_history: bool = False


@dataclass
class Dataset:
    name: str = "mnist"
    batch_size: int = 512
    val_split: float = 0.10
    data_dir: str | None = "/scratch/jmb1174/tensorflow_datasets"
    cache: bool = True


@dataclass
class Loss:
    acc: str = "classify"
    loss: str = "bce"


@dataclass
class Config:

    runs: int = 1
    run_i: int = 0
    net: Network = field(default_factory=Network)

    train: Train = field(default_factory=Train)
    dataset: Dataset = field(default_factory=Dataset)
    loss: Loss = field(default_factory=Loss)

    name: str = field(default_factory=lambda: f"{unique_id(4)}_{epoch_time(2)}")
    x64: bool = False  # whether to use 64 bit precision in jax

    platform: Union[str, None] = None  # gpu or cpu, None will let jax default
    # output_dir: str = './results/${hydra.job.name}'  # where to save results, if None nothing is saved

    seed: int = 1
    debug_nans: bool = False  # whether to debug nans
    # set advanced flags https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#nccl-flags
    advanced_flags: bool = True
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
    "run": {"dir": "presults/${dataset.name}/single/${name}"},
    "sweep": {"dir": "presults/${dataset.name}/multi/${name}"},
    # "mode": get_mode(),
    "sweeper": {"params": {**SWEEP}},
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {**SLURM_CONFIG},
    "job": {
        "env_set": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_PLATFORM_NAME": "cuda",
            "XLA_FLAGS": "--xla_gpu_force_compilation_parallelism=4",
        }
    },
    "job_logging": {"root": {"level": "WARN"}},
}


##################################
## problem wise default configs ##
##################################


cs = ConfigStore.instance()
cs.store(name="default", node=Config)

mnist_config = Config(
    dataset=Dataset(name="mnist"),
    train=Train(epochs=25),
    net=Network(arch="mlp", features=[15, 15, 15, 15]),
)

cifar10_config = Config(
    dataset=Dataset(name="cifar10"),
    train=Train(epochs=50),
    net=Network(
        arch="cnn",
        features=[64, 64, 64, 64, 64],
        kernel_size=3,
        pool=True,
        # norm_layer="layer",
        dropout=0.2,
    ),
)


mnist_l1_config = Config(
    dataset=Dataset(name="mnist"),
    train=Train(epochs=25),
    net=Network(arch="mlp", features=[]),
)

cs.store(name="mnist_l1", node=mnist_l1_config)
cs.store(name="mnist", node=mnist_config)
cs.store(name="cifar10", node=cifar10_config)
