import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from shrinkai.model.gradcam.gradcam import get_gradcam, plt_gradcam
import albumentations as A
import albumentations.pytorch.transforms as AT

class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        '''
        UnNormalizes an image given its mean and standard deviation
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        '''
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# Plot Gradcam

def plot_gradcam(target_layers, data, device, model, mean, std, class_names, misclassified=False):
    plt.style.use("dark_background")
    # logger.info('Plotting Grad-CAM...')

    
    # use the test images
    data, target = data.to(device), target.to(device) # Sending to Gradcam
    # get_gradcam
    gcam_layers, predicted_probs, predicted_classes = get_gradcam(
        data, target, model, device, target_layers)

    # get the denomarlization function
    unorm = UnNormalize(mean=mean, std=std)

    # If wrongly classified
    if misclassified: # Get 25 misclassified images
        for image, label, target in enumerate(zip(gcam_layers, predicted_probs, predicted_classes)):

            if label != target:
                in_data = zip(images)

    plt_gradcam(gcam_layers=gcam_layers, images=data, target_labels=target, predicted_labels= predicted_classes, class_labels= class_names, denormalize= unorm)


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

