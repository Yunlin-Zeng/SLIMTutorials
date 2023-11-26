import tensorflow as tf
import os

# Load the dataset
celeba = tf.keras.datasets.celeba
(data, labels), (test_data, test_labels) = celeba.load_data()

# Option 1: Save as HDF5
import h5py
with h5py.File('celeba.h5', 'w') as f:
    f.create_dataset('data', data=data)
    f.create_dataset('labels', data=labels)



