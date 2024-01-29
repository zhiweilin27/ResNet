import torch
from DataReader import *
from DataProcessing import *
from Model import Cifar

import os
import argparse

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--window_x", type=int, default=32, help="Number of past days to consider for input features (x)")
    parser.add_argument("--window_y", type=int, default=5, help="Number of future days for predicting returns (y)")
    parser.add_argument("--resnet_version", type=int, default=2, help="the version of ResNet")
    parser.add_argument("--resnet_size", type=int, default=5, 
                        help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=16, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=3, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=16, help='number of classes')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    ### YOUR CODE HERE
    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    open_dir = "/content/opens_data.csv"
    close_dir="/content/close_data.csv"
    high_dir="/content/high.csv"
    low_dir = "/content/low.csv"
    volume_dir = "/content/volume.csv"
    returns_dir = "/content/returns_data.csv" 
    

    ### YOUR CODE HERE
    x_train, y_train, x_test, y_test = load_data(open_dir,close_dir,high_dir,low_dir,volume_dir,returns_dir,split=0.9)
        
    model = Cifar(config).cuda()

    ### YOUR CODE HERE
    model.train(x_train, y_train, 200)
    model.test(x_test, y_test, [180, 190, 200])


    ### END CODE HERE

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)
