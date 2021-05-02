import torch
# import torch.nn as nn  # Loss function in nn
# import torch.nn.functional as F  # if needed
import numpy as np
import matplotlib.pyplot as plt
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
import json


class Trainer:
    def __init__(self, model, arg_path, optimizer, loss_func):
        self.model = model
        self.args = json.load(arg_path)
        self.optimizer = optimizer
        self.loss_func = loss_func

    def train(self, dset):
        train_args = self.args["train"]

    def test(self, dset):
        test_args = self.args["test"]

    @staticmethod
    def accuracy_eval(state, y, y_pred):
        if state == "train":
            print("Training metrics")
        else:
            print("Evaluation metrics")
