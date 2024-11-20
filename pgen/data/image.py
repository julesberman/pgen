import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

import pgen.io.result as R
from pgen.config import Dataset


def prepare_ds(preprocess_fn, batch_size, ds, cache=True, drop_remainder=False):
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
        list(ds.as_numpy_iterator())
    ds = ds.shuffle(buffer_size=ds.cardinality())
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


def get_simple_image_ds(d_cfg: Dataset):
    (full_train_set, test_dataset), ds_info = tfds.load(
        d_cfg.name,
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=d_cfg.data_dir,
    )

    num_classes = ds_info.features["label"].num_classes
    # R.RESULT["ds_info"] = ds_info
    # R.RESULT["num_classes"] = num_classes
    bs = d_cfg.batch_size

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    n_total = full_train_set.cardinality().numpy()

    train_ds_size = n_total * (1 - d_cfg.val_split)
    train_dataset = full_train_set.take(train_ds_size)
    val_dataset = full_train_set.skip(train_ds_size)

    train_dataset = prepare_ds(
        preprocess,
        bs,
        train_dataset,
        cache=d_cfg.cache,
        drop_remainder=d_cfg.drop_remainder,
    )
    val_dataset = prepare_ds(
        preprocess,
        bs,
        val_dataset,
        cache=d_cfg.cache,
        drop_remainder=d_cfg.drop_remainder,
    )
    test_dataset = prepare_ds(
        preprocess,
        bs,
        test_dataset,
        cache=d_cfg.cache,
        drop_remainder=d_cfg.drop_remainder,
    )

    # make dummy dataset to check just one example batch
    dummy_dataset = prepare_ds(preprocess, bs, full_train_set.take(bs), cache=False)
    (image, label) = next(iter(dummy_dataset))
    in_shape = image.shape
    out_shape = num_classes

    return (train_dataset, val_dataset, test_dataset), in_shape, out_shape
