import torch
"""
define the learning rate schedulers
"""


def get_lr_scheduler(lr_scheduler_name='Exponential', opt=None, **kwargs):
    lr_scheduler_lower = lr_scheduler_name.lower()
    lr_scheduler = None

    if lr_scheduler_lower == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt,
            **kwargs,
        )

    return lr_scheduler
