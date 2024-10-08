import math
import torch

class CosineAnnealingWithWarmupAndEtaMin(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, warmup_steps=0):
        """
        Custom Learning Rate Scheduler that combines Cosine Annealing with Warmup and minimum learning rate (eta_min).

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default is 0.
            last_epoch (int): The index of the last epoch. Default is -1.
            warmup_steps (int): Number of warmup steps. Default is 0.
        """
        if T_max <= warmup_steps:
            raise ValueError("T_max should be greater than warmup_steps.")
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super(CosineAnnealingWithWarmupAndEtaMin, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            return [(self.last_epoch + 1) / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_decay = (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps))) / 2
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]