import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class RepeatedDataset:
    """
    Wrapper class around a tf.data.Dataset object
    An iterator that can repeat the dataset indefinitely.
    Looping over the iterator gives the entire dataset once.
    """
    
    def __init__(self, dataset: tf.data.Dataset):
        self.dataset_length = len(dataset)
        self.iterator = dataset.repeat().as_numpy_iterator()
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.current_index < self.dataset_length:
            self.current_index += 1
            return next(self.iterator)
        self.current_index = 0
        raise StopIteration
    

class Dataset():
    """
    Wrapper class around two tf.data.Dataset objects, one for training and one for testing.
    Assumes the datasets are preprocessed.
    Handles data info, augmentation and folding.
    """
    
    test_ds: tf.data.Dataset = None
    train_val_ds: tf.data.Dataset = None
    
    k_fold_percentage: int = None
    input_shape: tuple = None
    num_classes: int = None
    
    current_batch_size: int = None
    current_train_size: int = None
    
    current_train_iterator: RepeatedDataset = None
    current_val_iterator: RepeatedDataset = None
    
    augmentation_layers: list[tf.keras.layers.Layer] = None
    
    
    def __init__(
        self,
        train_val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        num_folds: int,
        input_shape: tuple,
        num_classes: int
        ) -> None:
        
        self.train_val_ds = train_val_ds
        self.test_ds = test_ds
        
        self.k_fold_percentage = 100 // num_folds
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Initialize with an identity layer, such that the pipeline works even without augmentation layers
        self.augmentation_layers = [tf.keras.layers.Identity()]
        
        # Prepare dataset with the first fold, 1 epoch and a batch size of 4
        self.prepare_dataset(-1, 128)
    
    
    def add_augmentation_layer(self, layer: tf.keras.layers.Layer) -> None:
        self.augmentation_layers.append(layer)
    

    def get_input_dimensions(self) -> tuple:
        return self.input_shape
    

    def get_output_dimensions(self) -> tuple:
        return self.num_classes
    
    
    def get_train_size(self) -> int:
        return self.current_train_size

    
    def get_train_iterator(self) -> tf.data.Dataset:
        return self.current_train_iterator
    
    
    def get_val_iterator(self) -> tf.data.Dataset:
        return self.current_val_iterator
    
    
    def create_tf_dataset(self, ds: tf.data.Dataset, batch_size: int, train: bool) -> tf.data.Dataset:
        def __augment_dataset(input, label):
            augmentation = tf.keras.Sequential(self.augmentation_layers)
            input = augmentation(input, training=True)
            return input, label
        
        ds = ds.cache()
        if train:
            ds = ds.map(__augment_dataset, num_parallel_calls=tf.data.AUTOTUNE)
            
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds
        
    
    def prepare_dataset(self, fold: int, batch_size: int) -> None:
        """
        Set the fold of the dataset.
        Fold -1 trains on the complete training dataset and validates the test set.
        """
        if fold == -1:            
            current_train_ds = self.train_val_ds
            current_val_ds = self.test_ds
            
        else:
            start_percentage = fold * self.k_fold_percentage
            end_percentage = (fold + 1) * self.k_fold_percentage
            breakpoint_1 = len(self.train_val_ds) * start_percentage // 100
            breakpoint_2 = len(self.train_val_ds) * end_percentage // 100

            current_train_ds = self.train_val_ds.take(breakpoint_1).concatenate(self.train_val_ds.skip(breakpoint_2))
            current_val_ds = self.train_val_ds.skip(breakpoint_1).take(breakpoint_2 - breakpoint_1)
            self.current_batch_size = batch_size
            
        self.current_train_size = len(current_train_ds)
        self.current_train_iterator = RepeatedDataset(self.create_tf_dataset(current_train_ds, batch_size, train=True))        
        self.current_val_iterator = RepeatedDataset(self.create_tf_dataset(current_val_ds, batch_size, train=False))

