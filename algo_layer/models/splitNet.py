from torch import nn

batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
