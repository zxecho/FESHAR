import random

import numpy as np
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def frozen_net(self, models, frozen):
    for model in models:
        for param in self.global_net[model].parameters():
            param.requires_grad = not frozen
        if frozen:
            self.global_net[model].eval()
        else:
            self.global_net[model].train()


def add_gaussian_noise(tensor, mean, std):
    return torch.randn(tensor.size()) * std + mean


class AvgMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = 0
        self.avg = 0.

    def update(self, val, n=1):
        assert n > 0
        self.val += val
        self.n += n
        self.avg = self.val / self.n

    def get(self):
        return self.avg


class BestMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = -1

    def update(self, val, n):
        assert n > self.n
        if val > self.val:
            self.val = val
            self.n = n

    def get(self):
        return self.val, self.n
