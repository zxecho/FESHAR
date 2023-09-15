import argparse
import os
import numpy as np
import math

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, n_classes, input_channels, img_size, latent_dim, norm='gn'):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        if norm == 'bn':
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, input_channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )
        elif norm == 'gn':
            self.conv_blocks = nn.Sequential(
                nn.GroupNorm(128, 128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.GroupNorm(128, 128),
                nn.GELU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.GroupNorm(64, 64),
                nn.GELU(),
                nn.Conv2d(64, input_channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, input_channels, img_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, norm='gn'):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.GELU(), nn.Dropout2d(0.25)]
            if norm == 'bn':
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            elif norm == 'gn':
                block.append(nn.GroupNorm(out_filters, out_filters))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(input_channels, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = math.ceil(img_size / 2 ** 4)

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        # Forward pass of the convolutional blocks
        out = self.conv_blocks(img)
        # Reshape the output of the convolutional blocks to a vector
        out = out.view(out.shape[0], -1)
        # Forward pass of the adversarial layer
        validity = self.adv_layer(out)
        # Forward pass of the auxiliary layer
        label = self.aux_layer(out)

        # Return the validity and label
        return validity, label
