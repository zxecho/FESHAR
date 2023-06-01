import torch.nn as nn


def get_loss_function(loss_func_name):
    loss_fc = None
    if loss_func_name == 'cross_entropy':
        loss_fc = nn.CrossEntropyLoss()

    return loss_fc
