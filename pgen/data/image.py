import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

import pgen.io.result as R
from pgen.config import Dataset


def count_dataset(ds):
    return tf.data.experimental.cardinality(ds).numpy()


def prepare_ds(preprocess_fn, total_size, batch_size, ds, cache=True):
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
        list(ds.as_numpy_iterator())
    ds = ds.shuffle(buffer_size=total_size)
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=False)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)


def get_simple_image_ds(d_cfg: Dataset):
    (full_train_set, test_dataset), ds_info = tfds.load(
        d_cfg.name,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir=d_cfg.data_dir
    )

    num_classes = ds_info.features['label'].num_classes
    R.RESULT['ds_info'] = ds_info
    R.RESULT['num_classes'] = num_classes

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.
        label = tf.cast(tf.one_hot(label, depth=num_classes), tf.float32)
        return image, label

    n_total = count_dataset(full_train_set)

    train_ds_size = n_total * (1 - d_cfg.val_split)
    train_dataset = full_train_set.take(train_ds_size)
    val_dataset = full_train_set.skip(train_ds_size)

    n_train = count_dataset(train_dataset)
    n_val = count_dataset(val_dataset)
    n_test = count_dataset(test_dataset)

    train_dataset = prepare_ds(
        preprocess, n_total, d_cfg.batch_size, train_dataset, cache=d_cfg.cache)
    val_dataset = prepare_ds(
        preprocess, n_val, d_cfg.batch_size, val_dataset, cache=d_cfg.cache)
    test_dataset = prepare_ds(preprocess, n_total,
                              d_cfg.batch_size, test_dataset, cache=d_cfg.cache)

    # make dummy dataset to check just one example batch
    dummy_dataset = prepare_ds(preprocess, d_cfg.batch_size,
                               d_cfg.batch_size, full_train_set.take(d_cfg.batch_size), cache=d_cfg.cache)
    (image, label) = next(iter(dummy_dataset))
    in_shape, out_shape = image.shape, label.shape

    print('n_total:', n_total)
    print('n_train:', n_train)
    print('n_val:', n_val)
    print('n_test:', n_test)
    R.RESULT['n_total'] = n_total
    R.RESULT['n_train'] = n_train
    R.RESULT['n_val'] = n_val
    R.RESULT['n_test'] = n_test

    print('in_shape:', in_shape)
    print('out_shape:', out_shape)
    R.RESULT['in_shape'] = in_shape
    R.RESULT['out_shape'] = out_shape

    return (train_dataset, val_dataset, test_dataset), in_shape, out_shape
