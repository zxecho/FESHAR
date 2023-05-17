
def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)