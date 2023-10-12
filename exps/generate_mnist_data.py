import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from exps.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)


# Allocate data to users
def generate_mnist(raw_data_path, dataset_name, num_clients, num_classes, niid=False, real=True, partition=None):
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
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

    trainset = torchvision.datasets.MNIST(
        root=raw_data_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=raw_data_path + "rawdata", train=False, download=True, transform=transform)
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
                                    niid, real, partition, balance=True)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, real, partition)


def generate_mnist4robot(raw_data_path, dataset_name, num_clients, num_classes, niid=False, real=True, partition=None):
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

    trainset = torchvision.datasets.MNIST(
        root=raw_data_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=raw_data_path + "rawdata", train=False, download=True, transform=transform)
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
    train_X, train_y, train_statistic = separate_data((train_dataset_image, train_dataset_label), num_clients, num_classes,
                                    niid, real, partition, balance=True, least_samples=300)
    # for test dataset
    test_X, test_y, test_statistic = separate_data((test_dataset_image, test_dataset_label), num_clients, num_classes,
                                    niid=False, real=False, partition=None, balance=True)

    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, real, partition)


if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # real = True if sys.argv[2] == "realworld" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    niid = True
    real = True
    partition = 'dir'

    num_clients = 100
    num_classes = 10
    raw_data_path = "mnist/"

    dataset_name = 'non_iid4robot'

    # generate_mnist(raw_data_path, dataset_name, num_clients, num_classes, niid, real, partition)
    generate_mnist4robot(raw_data_path, dataset_name, num_clients, num_classes, niid, real, partition)

