import torch
# import numpy as np
# import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss
from torch.nn import Module
from torch.optim import Optimizer
import tqdm
import time


class TextTrainer:
    def __init__(self, model: Module, optimizer: Optimizer, loss_func: _Loss, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device
        self.model.to(device)

    def train(self, dataloader):
        self.model.train()
        start = time.time()
        correct, total = 0, 0
        total_loss, best_acc = 0.0, 0.0
        tq = tqdm.tqdm(dataloader)
        # tq = dset
        for idx, data in enumerate(tq):
            y = data[0]
            x = data[1]
            offset = data[2]
            y_pred = self.model(x, offset)

            correct += (y_pred == y).sum().item()
            total += y.size(0)

            y = y.unsqueeze(1)           # For BCE Loss
            y = y.float()
            loss = self.loss_func(y_pred, y)
            total_loss += loss

            self.optimizer.zero_grad()     #
            loss.backward()
            # Gradient clipping with normalization at 0.1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            running_acc = (correct / total * 1.0) * 100
            tq.set_postfix(LOSS=loss.item(), RUNNING_ACC=running_acc)
        self.accuracy_eval("train", correct, total,
                           total_loss / (idx+1), time.time() - start)
        return total_loss / (idx + 1)

    def test(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0
        start = time.time()
        idx = 0
        with torch.no_grad():
            for data in tqdm.tqdm(dataloader, mininterval=2):
                idx += 1
                y, x, offset = data
                correct += (y_pred == y).sum().item()
                total += y.size(0)
                y = y.unsqueeze(1)  # For BCE Loss
                y = y.float()
                y_pred = self.model(y, offset)
                loss = self.loss_func(y_pred, y)

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
