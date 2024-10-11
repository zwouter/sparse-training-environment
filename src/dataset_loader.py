import tensorflow_datasets as tfds
import tensorflow as tf
import datasets as huggingface

from my_datasets import Dataset


def load_dataset(name: str, num_folds: int, seed: int=43):
    print(f"Loading {name} dataset")
    if name == 'mnist':
        return load_mnist(num_folds, seed)
    elif name == 'fashion_mnist':
        return load_fashion_mnist(num_folds, seed)
    elif name == 'higgs':
        return load_higgs(num_folds, seed)
    elif name == 'electricity':
        return load_electricity(num_folds, seed)
    elif name == 'cifar10':
        return load_cifar10(num_folds, seed)
    elif name == 'svhn':
        return load_svhn(num_folds, seed)
    

def load_mnist(num_folds: int, seed: int):
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)
    
    train_ds = train_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = Dataset(
        train_val_ds=train_ds,
        test_ds=test_ds,
        num_folds=num_folds,
        input_shape=[28, 28, 1],
        num_classes=10,
    )
    return ds


def load_fashion_mnist(num_folds: int, seed: int):
    train_ds, test_ds = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, shuffle_files=True)
    
    train_ds = train_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = Dataset(
        train_val_ds=train_ds,
        test_ds=test_ds,
        num_folds=num_folds,
        input_shape=[28, 28, 1],
        num_classes=10,
    )
    
    return ds


def load_higgs(num_folds: int, seed: int):
    train_ds = huggingface.load_dataset('jxie/higgs', split='train', trust_remote_code=True)
    test_ds = huggingface.load_dataset('jxie/higgs', split='test', trust_remote_code=True)
    
    train_ds = tf.data.Dataset.from_tensor_slices(dict(train_ds.to_dict())).map(lambda x: (x["inputs"], x['label']))
    test_ds  = tf.data.Dataset.from_tensor_slices(dict(test_ds.to_dict())).map(lambda x: (x["inputs"], x['label']))
        
    ds = Dataset(
        train_val_ds=train_ds,
        test_ds=test_ds,
        num_folds=num_folds,
        input_shape=[28],
        num_classes=2,
    )
    return ds


def load_electricity(num_folds: int, seed: int):
    # For some reason splitting using string api gives unexpected results. Therefore, we split this dataset manually later.
    train_test_ds = huggingface.load_dataset('inria-soda/tabular-benchmark', "clf_num_electricity", split='train', trust_remote_code=True)
    train_test_ds = train_test_ds.map(lambda x: {'inputs': [x[key] for key in x.keys() if key != 'class'], 'label': x['class']})
    train_test_ds = train_test_ds.to_tf_dataset(columns="inputs", label_cols="label", shuffle=True)
        
    def map_labels(features, labels):
        # 'labels' is a tensor with values "UP" or "DOWN"
        labels = tf.where(labels == "UP", 1, 0)
        return features, labels
    
    train_test_ds = train_test_ds.map(map_labels)

    split_point = int(len(train_test_ds) * 0.7)
    train_ds = train_test_ds.take(split_point)
    test_ds = train_test_ds.skip(split_point)
    
    ds = Dataset(
        train_val_ds=train_ds,
        test_ds=test_ds,
        num_folds=num_folds,
        input_shape=[7],
        num_classes=2,
    )
    return ds


def load_cifar10(num_folds: int, seed):
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, shuffle_files=True)
    
    train_ds = train_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = Dataset(
        train_val_ds=train_ds,
        test_ds=test_ds,
        num_folds=num_folds,
        input_shape=[32, 32, 3],
        num_classes=10,
    )
    
    ds.add_augmentation_layer(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    ds.add_augmentation_layer(tf.keras.layers.RandomTranslation(0.1, 0.1, seed=seed))

    return ds


def load_svhn(num_folds: int, seed: int):
    train_ds, test_ds = tfds.load('svhn_cropped', split=['train', 'test'], as_supervised=True, shuffle_files=True)
    
    train_ds = train_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(__cast_to_float, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = Dataset(
        train_val_ds=train_ds,
        test_ds=test_ds,
        num_folds=num_folds,
        input_shape=[32, 32, 3],
        num_classes=10,
    )
    
    ds.add_augmentation_layer(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    ds.add_augmentation_layer(tf.keras.layers.RandomTranslation(0.1, 0.1, seed=seed))

    return ds


def __cast_to_float(x, y):
    return tf.cast(x, tf.float32) / 255, y
