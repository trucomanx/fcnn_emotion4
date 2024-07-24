#!/usr/bin/python

import tensorflow as tf
import numpy as np

class DataAugmentationEncDecGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=32, augment_fn=None):
        '''
        data: Matrix of type Numpy array
        batch_size: Batch size
        augment_fn: Augmented function, receive a numpy array and return a numpy array of same shape.
        '''
        self.data = data
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.indices = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]

        if self.augment_fn:
            batch_data = self.augment_fn(batch_data)

        return batch_data, batch_data  # Encoder-decoder model

    def on_epoch_end(self):
        np.random.shuffle(self.indices)



from sklearn.preprocessing import LabelBinarizer

class DataAugmentationClsGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size=32, augment_fn=None):
        '''
        data: Matrix of type Numpy array
        labels: Vector com as labels
        batch_size: Batch size
        augment_fn: Augmented function, receive a numpy array and return a numpy array of same shape.
        '''
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.indices = np.arange(len(self.data))
        
        # One-hot encode the labels
        self.label_binarizer = LabelBinarizer()
        self.one_hot_labels = self.label_binarizer.fit_transform(self.labels)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[batch_indices]
        batch_labels = self.one_hot_labels[batch_indices]

        if self.augment_fn:
            batch_data = self.augment_fn(batch_data)

        return batch_data, batch_labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


