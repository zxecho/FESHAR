import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16


class LocalModel(nn.Module):
    def __init__(self, base, predictor):
        super(LocalModel, self).__init__()

        self.base = base
        self.predictor = predictor

    def forward(self, x):
        out = self.base(x)
        out = self.predictor(out)

        return out


class FedAvgMLP(nn.Module):
    def __init__(self, in_features=13, num_classes=2, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
