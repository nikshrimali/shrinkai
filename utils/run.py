from shrinkai.model.gradcam.gradcam import get_gradcam, plt_gradcam
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from torch_lr_finder import LRFinder

# Importing model modules
from shrinkai.model.model_test import model_testing
from shrinkai.model.model_train import model_training
from shrinkai.model.gradcam import gen_gradcam

# Importing data process modules
from shrinkai.data_process.albumentation import *
from shrinkai.data_process.misclassified_data import *
from shrinkai.model.resnetmodel8 import ResNet18

import random, os
import numpy as np


import matplotlib.pyplot as plt

import yaml

def load_config(filename: str) -> dict:
    '''Load a configuration file as YAML'''
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    return config
def get_attributes(module, name, config, *args):
    '''Creates an instance of constructor as per the dict'''
    const_name = config[name]['type']
    return getattr(module, const_name)(*args, **config[name]['args'])


class Shrink:
    '''Shrinks the code and gets the output'''
    def __init__(self, in_config):
        self.config = in_config
        self.class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.mean = (0.491, 0.482, 0.446)
        self.std = (0.247, 0.243, 0.261)
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.model_path = self.config['modelpath']['args']

    plt.style.use("dark_background")

    def seed_everything(self,seed: int) -> None:
        '''Seeds the Code so that we get predictable outputs'''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


    def load_data(self, in_dir='./data'):
        '''Downloads the dataset and returns train and testloaders after applying the Transformations'''

        trainset = datasets.CIFAR10(in_dir, train=True, download=True, transform=cifar_alb_trainData())
        testset = datasets.CIFAR10(in_dir, train=False, download=True, transform=cifar_alb_testdata())

        self.trainloader = torch.utils.data.DataLoader(trainset, **self.config['train_data_loader']['args'])
        self.testloader = torch.utils.data.DataLoader(testset, **self.config['test_data_loader']['args'])
        return self.trainloader, self.testloader

    def mean_std_dev(self):
        pass

    def show_data(self, mode='train', n=25):
        '''Plots the images on a gridplot to show the images passed via dataloader'''

        figure = plt.figure(figsize=(20,20))
        images = None
        labels = None
        
        if mode.lower() == 'train':
            images, labels = next(iter(self.trainloader))
            labels = np.array(labels)
        elif mode.lower() == 'test':
            images, labels = next(iter(self.testloader))
            labels = np.array(labels)

        images = self.denormalize(images)
        for index in range(1,n+1):
            plt.subplot(5,5,index)
            plt.axis('off')
            # Gets the first n images of the dataset
            plt.imshow(np.transpose(images[index], (1, 2, 0))) # Plots the dataset
            plt.title(self.class_names[labels[index]])
        
    
    def get_batched_data(self,in_data):
        '''Takes in the list data and outputs data, targets and preds'''
        in_imgs = []
        in_preds = []
        in_targets = []
        
        for index, i in enumerate(in_data):
            in_imgs.append(i[0])
            in_preds.append(i[1])
            in_targets.append(i[2])
        return torch.stack(in_imgs), torch.stack(in_preds), torch.stack(in_targets)
   

    def plot_gradcam(self, target_layers, images, pred, target, nimgs):
        '''Plot GradCam - '''
        index = 0
        in_data = None
        # model.load_state_dict(torch.load(self.model_path))

        images = images[index:nimgs].to(self.device)
        target = target[index:nimgs]
        pred = pred[index:nimgs]

        gcam_layers, predicted_probs, predicted_classes = get_gradcam(images, target, self.model, self.device, target_layers)

        # get the denomarlization function
        unorm = UnNormalize(mean=self.mean, std=self.std)

        plt_gradcam(gcam_layers=gcam_layers, images=images, target_labels=target, predicted_labels= predicted_classes, class_labels= self.class_names, denormalize= unorm)
    
    def get_gradoutput(self, misclassified=False):
        '''Outputs a gradcam output when Inputting an image'''
        if misclassified:
            in_data = self.misclassified
        else:
            in_data = self.correct_classified

        target_layers = ["layer1", "layer2", "layer3", "layer4"]
        imgs, preds, targets = self.get_batched_data(in_data)
        self.plot_gradcam(target_layers, imgs, preds, targets, 25)


    def denormalize(self,tensor):
        '''Denormalize the data'''
        if not tensor.ndimension() == 4:
            raise TypeError('tensor should be 4D')

        mean = torch.FloatTensor(self.mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
        std = torch.FloatTensor(self.std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

        return tensor.mul(std).add(mean)
    
    def get_model(self, train=True):
        
        import shrinkai.model as model_arch
        self.model = get_attributes(model_arch, 'model', self.config).to(self.device)
        
        if train:
            '''Trains the model and sends the output'''
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), **self.config['optimizer']['args'])
            scheduler = ReduceLROnPlateau(optimizer, **config['ReduceLROnPlateau']['args'])
            # scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

            train_acc = []
            train_losses = []
            test_acc = []
            test_losses = []

            EPOCHS = self.config['training']['epochs']
            print(f'Starting Training for {EPOCHS} Epochs')
            try:
                os.remove(self.model_path) # deleting the existing file
            except:
                print('Existing trained model not removed')

            for i in range(EPOCHS):
                print(f'EPOCHS : {i}')
                model_training(self.model, self.device, self.trainloader, optimizer, train_acc, train_losses, l1_loss=False)
                torch.save(self.model.state_dict(), self.model_path)
                scheduler.step()
                self.misclassified, self.correct_classified = model_testing(self.model, self.device, self.testloader, test_acc, test_losses)
                # return self.model
        
    def test_model(self):
        '''Loads and saves the test model'''
        train_acc = []
        test_losses = []
        test_acc = []

        model_path = 'latest_model.h5'
        self.model.load_state_dict(torch.load(model_path))
        self.misclassified, self.correct_classified = model_testing(self.model, self.device, self.testloader, test_acc, test_losses)
        return self.misclassified, self.correct_classified

    def findbestlr(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr= 0.00001, momentum= 0.95, weight_decay= 0.0005)
        self.lr_finder = LRFinder(self.model, optimizer, criterion, device=self.device)
        print(self.config['range_test']['args'])
        self.lr_finder.range_test(self.trainloader, **self.config['range_test']['args'])
        self.lr_finder.plot() # to inspect the loss-learning rate graph
        self.lr_finder.reset() # to reset the model and optimizer to their initial state
        return self.lr_finder

