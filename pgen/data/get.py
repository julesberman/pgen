
import tensorflow as tf
import tensorflow_datasets as tfds

import pgen.io.result as R
from pgen.config import Config, Dataset
from pgen.data.image import get_simple_image_ds


def get_dataset(cfg: Config):

    datasets, in_shape, out_shape = get_simple_image_ds(cfg.dataset)

    return datasets, in_shape, out_shape
