
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import matplotlib.pyplot as plt
import wandb
import torch.nn as nn


class WarmupScheduler(nn.Module):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, base_lr, n_warmup_steps, after_scheduler):
        super().__init__()
        self._optimizer = optimizer
        self.base_lr = base_lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        self.after_scheduler = after_scheduler
        self.lr_history = []


    def step(self, step=None, loss=None):
        "Step with the inner optimizer"
        self._update_learning_rate(loss=loss)

    def _update_learning_rate(self, loss=None):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1

        if self.n_steps < self.n_warmup_steps:
            lr = self.base_lr

            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr

            self.lr_history.append(lr)
        else:
            # self.after_scheduler.step(self.n_steps - self.n_warmup_steps)
            self.after_scheduler.step(loss)
            for param_group in self._optimizer.param_groups:
                lr = param_group['lr']
            self.lr_history.append(lr)
        wandb.log({"lr":lr})




