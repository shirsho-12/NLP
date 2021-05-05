import torch
from torch.utils.data import dataset
from torchvision import datasets as image_dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchtext


def fashion_mst_data():
    """FashionMST Downloader (Refresher)"""
    train_data = image_dset.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    test_data = image_dset.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    return {"train": train_data, "test": test_data}


def amazon_polarity_data():
    """Amazon Reviews Polarity Dataset: 2 classes 0-1"""
    train_data, test_data = torchtext.datasets.AmazonReviewPolarity(root="data", split=("train", "test"))
    return {"train": train_data, "test": test_data}
