import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    x_train = np.zeros((0, 3072))  
    y_train = np.zeros((0,))
    x_test = np.zeros((0, 3072))
    y_test = np.zeros((0,))
    
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
                
        if os.path.isdir(file_path) or file_name.endswith("html") or file_name.endswith("meta"):
            continue  
        
        with open(file_path, 'rb') as fo:
            di = pickle.load(fo, encoding='bytes')
            x = np.array(di[b'data'])
            y = np.array(di[b'labels'])
            
            if file_name.startswith('test'):
                x_test = np.concatenate((x_test, x))
                y_test = np.concatenate((y_test, y))

            elif file_name.startswith('data'):
                x_train = np.concatenate((x_train, x),axis=0)
                y_train = np.concatenate((y_train, y),axis=0)

    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid