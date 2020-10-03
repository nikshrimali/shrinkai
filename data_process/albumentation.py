from shrinkai.data_process.transformations import *
from shrinkai.data_process.getdata import GetCIFAR10_TrainData
from shrinkai.data_process.misclassified_data import *
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

def cifar_alb_trainData():
    '''Apply Albumentations data transforms to the dataset and returns iterable'''
    mean = (0.491, 0.482, 0.446)
    std = (0.247, 0.243, 0.261)
    train_transform = [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(height=32, width=32),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(),
        A.ElasticTransform(),
        # A.MaskDropout((10,15), p=1),
        A.Cutout(num_holes=1,max_h_size=16,max_w_size=16,fill_value=mean ,always_apply=False, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()]

    transforms_result = A.Compose(train_transform)
    return lambda img:transforms_result(image=np.array(img))["image"]


def cifar_alb_testdata():
    mean = (0.491, 0.482, 0.446)
    std = (0.247, 0.243, 0.261)
    test_transform = [
        A.Normalize(mean=mean, std=std),
        ToTensorV2()]
    transforms_result = A.Compose(test_transform)
    return lambda img:transforms_result(image=np.array(img))["image"]
