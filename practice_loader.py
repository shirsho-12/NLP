import torch
from torch.utils.data import dataset
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def fashion_mst_data():
    """FashionMST Downloader (Refresher)"""
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    return {"train": train_data, "test": test_data}
