import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.models import resnet18

from algo_layer.models.model_utils import weights_init


class Extractor(nn.Module):

    def __init__(self, image_channel):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(image_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.extractor(x)
        return x


class Classifier(nn.Module):

    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Generator(nn.Module):

    def __init__(self, num_classes, noise_dim, feature_num):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, feature_num * 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, z, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        zy = torch.cat([z, y], 1)
        return self.generator(zy)


class Discriminator(nn.Module):

    def __init__(self, num_classes, feature_num, feature_size):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.feature_num = feature_num

        self.embedding = nn.Embedding(num_classes, num_classes)
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_num + num_classes, 128, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, f, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        y = y.expand(y.size(0), self.num_classes, self.feature_size, self.feature_size)
        fy = torch.cat([f, y], 1)
        return self.discriminator(fy).squeeze(-1).squeeze(-1)