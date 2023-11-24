from torchvision.transforms import transforms
import torchvision.datasets as datasets
import torch
import numpy as np
import PIL
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import utils.svhn_loader as svhn

class load_np_dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, targets_path, transform):
        self.data = np.load(imgs_path)
        self.targets = np.load(targets_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img , target = self.data[idx], self.targets[idx]
            
        img = PIL.Image.fromarray(img)
        img = self.transform(img)

        return img, target


def load_cifar10(cifar10_path):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = datasets.CIFAR10(
        cifar10_path, train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        cifar10_path, train=False, transform=test_transform, download=True)

    return train_data, test_data

def load_svhn(svhn_path, transform=None, include_extra=False):

    print('loading SVHN')
    if not include_extra:
        train_data = svhn.SVHN(root=svhn_path, split="train",
                                 transform=transform)
    else:
        train_data = svhn.SVHN(root=svhn_path, split="train_and_extra",
                               transform=transform)

    test_data = svhn.SVHN(root=svhn_path, split="test",
                              transform=transform)

    train_data.targets = train_data.targets.astype('int64')
    test_data.targets = test_data.targets.astype('int64')
    return train_data, test_data


def combiner(datasets):
    return ConcatDataset([dataset for dataset in datasets])


