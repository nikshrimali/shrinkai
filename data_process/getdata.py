# Code for downloading the data
from torchvision import datasets
from torchvision import transforms
import albumentations as A

class Get_MNISTTrainData:
    def __init__(self, dir_name, transformations):
        self.dirname = dir_name
        self.transformations = transformations
        self.train_transforms = transforms.Compose(self.transformations.trainparams())
        self.test_transforms = transforms.Compose(self.transformations.testparams())

    def download_train_data(self):
        return datasets.MNIST('./data', train=True, download=True, transform=self.train_transforms)

    def download_test_data(self):
        return datasets.MNIST('./data', train=False, download=True, transform=self.test_transforms)

class GetCIFAR10_TrainData:
    '''Downloads and returns training data
    Args:
    transforms: normal - Pytorch transformations or alb - Pytorch Albumentations
    Returns:
    dataset
    '''
    def __init__(self, dir_name='..'):
        self.dirname = dir_name
    def download_train_data(self, train_augmentations=None):
        # if train_augmentations is None:
        #     pass
        #     # if augmentations is "alb":
        #     #     self.train_transforms = transforms.Compose(self.transformations.trans_albumentations())
        #     #     self.test_transforms = transforms.Compose(self.transformations.testparams())
        #     # else:
        #     #     self.train_transforms = transforms.Compose(self.transformations.trainparams())
        #     #     self.test_transforms = transforms.Compose(self.transformations.testparams())
        # else:
        #     # train_transforms = train_augmentations
        transforms_result = A.Compose(train_augmentations)
        return lambda img:transforms_result(image=np.array(img))["image"] 

    def download_test_data(self, test_augmentations=None):
        if test_augmentations is None:
            pass
        else:
            test_transforms = test_augmentations

        return datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose(test_augmentations))

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
