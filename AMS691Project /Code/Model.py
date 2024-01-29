import os
import time
from numpy.random.mtrand import sample
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from DataReader import load_data
from attention_resnet import ResNet
#from NetWork import ResNet
from DataProcessing import *
import matplotlib.pyplot as plt


""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )

        ### YOUR CODE HERE
        # define cross entropy loss and optimizer

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optimizer, 80, 0.1)

        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        x_train, y_train = rolling_array(x_train, y_train, self.config.window_x, self.config.window_y)
        x_train, y_train = data_normalization_augmentation(x_train, y_train, training=True)
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size
        print('### Training... ###')
        losses = []
        epochs = []
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            curr_x_train = curr_x_train.transpose(0, 3, 1, 2)
            total_loss = 0.0
            ### YOUR CODE HERE
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
          
                start = i * self.config.batch_size
                end = start + self.config.batch_size
                if end > curr_x_train.shape[0]:
                    continue
                x_batch = []
                for j in range(start, end):
                    x_batch.append(curr_x_train[j])
                x_batch = np.array(x_batch)
                y_batch = curr_y_train[start:end]
                y_batch = np.array(y_batch)
                x_batch = torch.tensor(x_batch,dtype=torch.float32).cuda()
                y_batch = torch.tensor(y_batch,dtype=torch.long).cuda()
                
                output = self.network(x_batch)
                loss = self.criterion(output, y_batch)
                total_loss += loss.item()
       
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            self.scheduler_lr.step() 
            duration = time.time() - start_time
            loss = total_loss/num_batches
            losses.append(loss)
            epochs.append(epoch)
            plt.plot(epochs, losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss vs. Epoch')
            plt.savefig('loss_vs_epoch.png')

            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
                  

            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test(self, x, y, checkpoint_num_list):
        self.network.eval()
        x, y = rolling_array(x, y, self.config.window_x, self.config.window_y)   
        x, y = data_normalization_augmentation(x, y, training=False)
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpointfile)
            self = self.float()
            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                sample_x = x[i].transpose(2,0,1)
                sample_x = np.array(sample_x.reshape(1,6,32,32))
                sample_x = torch.tensor(sample_x,dtype=torch.float32).cuda()
                sample_pred = self.network(sample_x)
                preds.append(torch.argmax(sample_pred))
                ### END CODE HERE
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))

           
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))
