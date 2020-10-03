from torchvision import transforms
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensor
# from albumentations.pytorch import ToTensorV2


# Returns a list of transformations when called

class GetTransforms:
    '''Returns a list of transformations when type as requested amongst train/test
       Transforms('train') = list of transforms to apply on training data
       Transforms('test') = list of transforms to apply on testing data'''

    def __init__(self):
    
        self.mean = (0.491, 0.482, 0.446)
        self.std = (0.247, 0.243, 0.261)
    
    def trainparams(self):
        '''Applies PMDA's to the dataset'''

        train_transformations = [ #resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation(10),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize(self.mean, self.std) #Normalize all the images
            ]
        return train_transformations


    def testaugs(self):
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
        ]
        return test_transforms

def get_mean_std(dataset):

    train_data = dataset.data
    (train_data).shape
    # train_data = trainset.transform(train_data)
    train_data = train_data/255
    print(np.mean(train_data, axis = (0,1,2)))
    print(np.std(train_data, axis = (0,1,2)))

    train_tensor = torch.from_numpy(train_data)

    print(f'Mean of dataset {torch.mean(train_tensor)}')
    print(f'Mean of dataset {torch.std(train_tensor)}')
    # print(np.mean(trainset.data[0], axis = 2))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor