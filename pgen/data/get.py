import pgen.io.result as R
from pgen.config import Config, Dataset
from pgen.data.image import get_simple_image_ds


def get_dataset(cfg: Config):

    (train_ds, val_ds, test_ds), in_shape, out_shape = get_simple_image_ds(cfg.dataset)

    bs = cfg.dataset.batch_size
    info = {
        "n_train": len(train_ds) * bs,
        "n_val": len(val_ds) * bs,
        "n_test": len(test_ds) * bs,
        "in_shape": in_shape,
        "out_shape": out_shape,
    }

    R.RESULT.update(info)

    [print(f"{k}: {v}", end=" | ") for k, v in info.items()]
    print("\n")

    return (train_ds, val_ds, test_ds), in_shape, out_shape
