import torch
import numpy as np
import math


class RampupWeight:
    def __init__(self, end_epoch: int, start_epoch: int = 0, slope: float = 5):
        assert end_epoch > start_epoch
        assert slope > 0

        self.slope = float(slope)
        self.start_epoch = float(start_epoch)
        self.end_epoch = float(end_epoch)

    def __call__(self, epoch):
        epoch = float(epoch)

        if epoch < self.start_epoch:
            return 0.0
        elif epoch > self.end_epoch:
            return 1.0
        else:
            t = (epoch - self.start_epoch) / \
                (self.end_epoch - self.start_epoch)
            return np.exp(-self.slope*(1-t)**2)


class CosineAnnealingWithRestartsLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1):
        self.T_max = T_max
        self.T_mult = T_mult
        self.restart_every = T_max
        self.eta_min = eta_min
        self.restarts = 0
        self.restarted_at = 0
        super().__init__(optimizer, last_epoch)

    def restart(self):
        self.restart_every *= self.T_mult
        self.restarted_at = self.last_epoch

    def cosine(self, base_lr):
        return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.step_n / self.restart_every)) / 2

    @property
    def step_n(self):
        return self.last_epoch - self.restarted_at

    def get_lr(self):
        if self.step_n >= self.restart_every:
            self.restart()
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


class CosineAnnealingLRFlatEnd(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, 
                 T_max, 
                 eta_min=0, 
                 last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_max:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min for base_lr in self.base_lrs]
