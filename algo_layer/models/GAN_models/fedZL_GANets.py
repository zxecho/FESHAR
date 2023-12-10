import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision.models import resnet18

from algo_layer.models.model_utils import weights_init


class Extractor(nn.Module):

    def __init__(self, image_channel, feature_in_dim, feature_out_dim):
        super(Extractor, self).__init__()
        # self.extractor = nn.Sequential(
        #     nn.Conv2d(image_channel, 6, 5),
        #     nn.AvgPool2d(2, 2),
        #     nn.Sigmoid(),
        #     nn.Conv2d(6, 16, 5),
        #     nn.AvgPool2d(2, 2),
        #     nn.Sigmoid(),
        # )

        self.extractor = nn.Sequential(
            nn.Conv2d(image_channel,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(feature_in_dim, feature_out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Classifier(nn.Module):

    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Generator(nn.Module):

    def __init__(self, num_classes, noise_dim, feature_num):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + num_classes, 256, 2, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 2, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(256, 128, 2, 1, 0, bias=False),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, feature_num * 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, z, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        zy = torch.cat([z, y], 1)
        return self.generator(zy)

class Generative_model(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian

        y_input = torch.nn.functional.one_hot(labels, self.num_classes).to(self.device)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z

class Discriminator(nn.Module):

    def __init__(self, num_classes, feature_size, feature_num):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.embedding = nn.Embedding(num_classes, num_classes)
        self.discriminator = nn.Sequential(
            spectral_norm(nn.Conv2d(feature_num + num_classes, 128, 2, 1, 0, bias=False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 2, 1, 0, bias=False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Conv2d(256, 512, 2, 1, 0, bias=False)),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, 2, 1, 0, bias=False)),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, f, y):
        y = self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        y = y.expand(y.size(0), self.num_classes, self.feature_size, self.feature_size)
        fy = torch.cat([f, y], 1)
        return self.discriminator(fy).squeeze(-1).squeeze(-1)