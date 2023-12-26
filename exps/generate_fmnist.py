# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from exps.dataset_utils import check, separate_data, split_data, save_file, save_each_file

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "fmnist/"


# Allocate data to users
def generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


def generate_fmnist4robot(raw_data_path, dataset_name, num_clients, num_classes, niid=False, real=True, partition=None):
    dir_path = raw_data_path + dataset_name + "/"
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, real, partition):
    #     return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    train_dataset_image = trainset.data.cpu().detach().numpy()
    train_dataset_label = trainset.targets.cpu().detach().numpy()

    test_dataset_image = testset.data.cpu().detach().numpy()
    test_dataset_label = testset.targets.cpu().detach().numpy()

    # for training dataset
    train_X, train_y, train_statistic = separate_data((train_dataset_image, train_dataset_label), num_clients,
                                                      num_classes,
                                                      niid, real, partition, balance=False)
    # for test dataset
    test_X, test_y, test_statistic = separate_data((test_dataset_image, test_dataset_label), num_clients, num_classes,
                                                   niid=False, real=False, partition=None, balance=True)

    def split_each_data(X_train, y_train, X_test, y_test):
        # Split dataset
        train_data, test_data = [], []
        num_samples = {'train': [], 'test': []}

        # 该client的标签个数
        n_c = len(y_train)
        for i in range(n_c):
            train_data.append({'x': X_train[i], 'y': y_train[i]})
            num_samples['train'].append(len(y_train[i]))
            test_data.append({'x': X_test[i], 'y': y_test[i]})
            num_samples['test'].append(len(y_test[i]))

        print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
        print("The number of train samples:", num_samples['train'])
        print("The number of test samples:", num_samples['test'])
        print()
        del X_train, y_train, X_test, y_test
        # gc.collect()

        return train_data, test_data

    train_data, test_data = split_each_data(train_X, train_y, test_X, test_y)

    # save train dataset and config
    save_each_file(config_path, train_path, train_data, num_clients, num_classes, train_statistic,
                   niid, real, partition, 'train')

    # save test dataset and config
    save_each_file(config_path, test_path, test_data, num_clients, num_classes, test_statistic,
                   niid, real, partition, 'test')


if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None
    #
    # generate_fmnist(dir_path, num_clients, num_classes, niid, balance, partition)

    niid = True
    real = True
    partition = 'dir'

    num_clients = 20
    num_classes = 10
    raw_data_path = "mnist/"

    dataset_name = 'non_iid4robot'

    # generate_mnist(raw_data_path, dataset_name, num_clients, num_classes, niid, real, partition)
    generate_fmnist4robot(raw_data_path, dataset_name, num_clients, num_classes, niid, real, partition)