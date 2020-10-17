from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import hashlib
import gzip
import errno
import tarfile
import zipfile
import glob
from shrinkai.data_process.albumentation import resnet_train_alb, resent_test_alb


class TinyImageNet(Dataset):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    dataset_folder_name = 'tiny-imagenet-200'

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download and (not os.path.isdir(os.path.join(self.root, self.dataset_folder_name))):
            self.download_data()
        self.split_dir = 'train' if train else 'val'
        self.split_dir = os.path.join(
            self.root, self.dataset_folder_name, self.split_dir)
        self.image_paths = sorted(glob.iglob(os.path.join(
            self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))

        self.target = []
        self.labels = {}

        # build class label - number mapping
        with open(os.path.join(self.root, self.dataset_folder_name, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip()
                                       for text in fp.readlines()])
        self.label_text_to_number = {
            text: i for i, text in enumerate(self.label_texts)}

        # build labels for NUM_IMAGES_PER_CLASS images
        if train:
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels[f'{label_text}_{cnt}.{self.EXTENSION}'] = i

        # build the validation dataset
        else:
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        self.target = [self.labels[os.path.basename(
            filename)] for filename in self.image_paths]

    def download_data(self):
        download_and_extract_archive(self.url, self.root, filename=self.filename)

    def __getitem__(self, index):
        filepath = self.image_paths[index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.target[index]

        if self.transform:        
            img = self.transform(img)

        if self.target_transform:
            print('Applying transforms')
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_paths)

def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    extract_archive(archive, extract_root, remove_finished)

def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath)
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.'
                  ' Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath)
        else:
            raise e

def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, 'r:xz') as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(
            os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)

def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")

def get_imagenet_loader(train_augmentations, test_augmentations, train_arguments, test_arguments):
    train_set = TinyImageNet(
            '..',
            train=True,
            download=True,
            transform=train_augmentations())
        
    test_set = TinyImageNet(
            '..',
            train=False,
            download=True,
            transform=test_augmentations())
    
    trainloader = DataLoader(train_set, **train_arguments)
    testloader = DataLoader(test_set, **test_arguments)
    return trainloader, testloader
