import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm
import time


class TextTrainer:
    def __init__(self, model: Module, optimizer: Optimizer, loss_func: _Loss, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.model.to(device)

    def train(self, dset):
        self.model.train()
        start = time.time()
        loss, running_acc = 0, 0
        correct, total = 0, 0
        total_loss, best_acc = 0.0, 0.0
        for idx, data in tqdm(enumerate(dset), loss=loss, running_acc=running_acc):
            y = data[0]
            x = data[1]
            offset = data[2]
            y_pred = self.model(x, offset)
            loss = self.loss_func(y_pred, y)
            total_loss += loss
            self.optimizer.zero_grad()     ##
            loss.backward()
            # Gradient clipping with normalization at 0.1
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.1)
            self.optimizer.step()

            correct += (y_pred == y).sum().item().cpu()
            total += y.size(0)
            running_acc = (correct / total * 1.0) * 100
        self.accuracy_eval("train", correct, total,
                           total_loss / (idx+1), time.time() - start)
        return total_loss / (idx + 1)

    def test(self, dset):
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0
        start = time.time()
        with torch.no_grad():
            for idx, data in tqdm(enumerate(dset)):
                y, x, offset = data
                y_pred = self.model(y, offset)
                loss = self.loss_func(y_pred, y)
                correct += (y_pred == y).sum().item().cpu()
                total += y.size(0)
                total_loss += loss
        self.accuracy_eval("eval", correct, total,
                           total_loss / (idx + 1), time.time() - start)
        return correct * 1.0 / total

    @staticmethod
    def accuracy_eval(state, correct, total, loss, time_taken):
        if state == "train":
            print("Training:")

        else:
            print("Evaluation:")
        print(f"Accuracy: {correct * 100.0 / total} \t Loss: {loss} "
              f"\t Time taken: {time_taken:.2f}")