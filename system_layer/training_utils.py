
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
