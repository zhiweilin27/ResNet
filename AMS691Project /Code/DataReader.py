import os
import pickle
import numpy as np

def load_data(open_dir,close_dir,high_dir,low_dir,volume_dir,returns_dir,split=0.9):

    open_data = np.genfromtxt(open_dir, delimiter=',', skip_header=1)
    close_data = np.genfromtxt(close_dir, delimiter=',', skip_header=1)
    volume_data = np.genfromtxt(volume_dir, delimiter=',', skip_header=1)
    high_data = np.genfromtxt(high_dir, delimiter=',', skip_header=1)
    low_data = np.genfromtxt(low_dir, delimiter=',', skip_header=1)
    return_data = np.genfromtxt(returns_dir, delimiter=',', skip_header=1)

    open_channel = open_data[:, 1:]
    close_channel = close_data[:, 1:]
    high_channel = high_data[:, 1:]
    low_channel = low_data[:, 1:]
    volume_channel = volume_data[:, 1:]
    return_channel = return_data[:, 1:] 
    x = np.dstack((open_channel, close_channel,high_channel, low_channel, volume_channel,return_channel))
    
    y = return_data[:, 1:]    
    split_index = int(len(x) * split)
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return x_train, y_train, x_test, y_test

def train_valid_split(x_train, y_train, split=0.9):
    split_index = int(len(x_train) * split)
    x_train_new, x_valid = x_train[:split_index], x_train[split_index:]
    y_train_new, y_valid = y_train[:split_index], y_train[split_index:]
    
    return x_train_new, y_train_new, x_valid, y_valid