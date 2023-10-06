import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch
from system_layer.training_utils import weights_init


class Extractor(nn.Module):

    def __init__(self, input_channels):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.AvgPool2d(2, 2),
            nn.Sigmoid(),
            nn.Conv2d(6, 16, 5),
            nn.AvgPool2d(2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.extractor(x)
        return x


class Classifier(nn.Module):

    def __init__(self, n_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Generator(nn.Module):

    def __init__(self, n_classes, latent_dim, feature_num):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(n_classes, n_classes)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + n_classes, 256, 2, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(512, 256, 2, 1, 0, bias=False),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, feature_num * 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, z, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        zy = torch.cat([z, y], 1)
        return self.generator(zy)


class Conditional_D(nn.Module):

    def __init__(self, n_classes, feature_num, feature_size):
        super(Conditional_D, self).__init__()
        self.num_classes = n_classes
        self.feature_size = feature_size
        self.embedding = nn.Embedding(n_classes, n_classes)
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_num + n_classes, 128, 2, 1, 0, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 2, 1, 0, bias=False)),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Conv2d(256, 512, 2, 1, 0, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, 2, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, f, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        y = y.expand(y.size(0), self.num_classes, self.feature_size, self.feature_size)
        fy = torch.cat([f, y], 1)
        return self.discriminator(fy).squeeze(-1).squeeze(-1)


class Discriminator(nn.Module):

    def __init__(self, n_classes, feature_num, feature_size):
        super(Discriminator, self).__init__()
        self.num_classes = n_classes
        self.feature_size = feature_size
        self.embedding = nn.Embedding(n_classes, n_classes)
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_num, 128, 2, 1, 0, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 2, 1, 0, bias=False)),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Conv2d(256, 512, 2, 1, 0, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, 2, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, f):
        return self.discriminator(f).squeeze(-1).squeeze(-1)