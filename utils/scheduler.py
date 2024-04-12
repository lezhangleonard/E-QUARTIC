import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsWithDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False, decay_factor=0.9):
        self.decay_factor = decay_factor
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
        
        if last_epoch != -1:
            self.eta_max *= (self.decay_factor ** (self.last_epoch // self.T_0))

    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs
        elif self.last_epoch < self.T_max:
            self.eta_max = self.base_lrs[0] * (self.decay_factor ** (self.last_epoch // self.T_0))
            return [self.eta_min + (base_lr * self.decay_factor ** (self.last_epoch // self.T_0) - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
        return [self.eta_min + (self.eta_max - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
