from shrinkai.data_process.transformations import *
# from shrinkai.data_process.getdata import GetCIFAR10_TrainData
from shrinkai.data_process.misclassified_data import *
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensor

import numpy as np
import torch

import cv2

mean = (0.491, 0.482, 0.446)
std = (0.247, 0.243, 0.261)

def cifar_alb_trainData():
    '''Apply Albumentations data transforms to the dataset and returns iterable'''

    train_transform = [
        A.HorizontalFlip(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.25),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.RandomGamma(p=0.25),  
        A.CLAHE(p=0.25),
        A.ChannelShuffle(p=0.1),
        A.ElasticTransform(p=0.1),
        A.MotionBlur(blur_limit=17, p=0.1),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=mean, always_apply=False, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensor()]

    transforms_result = A.Compose(train_transform)
    return lambda img:transforms_result(image=np.array(img))["image"]

def cifar_alb11():
    '''Applies image augmentations to image dataset 
    RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    
    Returns:
        list of transforms'''
    mean = (0.491, 0.482, 0.446)
    mean = np.mean(mean)
    train_transforms = [
        A.Normalize(mean=mean, std=std),
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=4, always_apply=True, p=1.0),
        A.RandomCrop (32, 32, always_apply=True, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=mean, always_apply=False, p=0.5),
        ToTensor()
    ]
    transforms_result = A.Compose(train_transforms)
    return lambda img:transforms_result(image=np.array(img))["image"]

def cifar_alb_testdata():
    mean = (0.491, 0.482, 0.446)
    std = (0.247, 0.243, 0.261)
    test_transform = [
        A.Normalize(mean=mean, std=std),
        ToTensor()]
    transforms_result = A.Compose(test_transform)
    return lambda img:transforms_result(image=np.array(img))["image"]
