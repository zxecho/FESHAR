import numpy as np
import os
import sys
import random
import torch
import deeplake
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from exps.dataset_utils import check, separate_data, split_data, save_file, save_each_file

random.seed(1)
np.random.seed(1)


def generate_fer2013(raw_data_path, dataset_name, num_clients, num_classes, niid=False, balance=True, partition=None):
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
    transform_train = transforms.Compose([transforms.Resize(32),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FER2013(root=raw_data_path + "rawdata", split='train', transform=transform_train)

    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    for data, label in trainloader:
        trainset_data = data
        trainset_label = label

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset_data.cpu().detach().numpy())
    dataset_label.extend(trainset_label.cpu().detach().numpy())

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


def generate_fer2013_for_robot(raw_data_path, dataset_name, num_clients, num_classes,
                               niid=False, balance=True, partition=None):
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
    transform_train = transforms.Compose([transforms.Resize(32),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FER2013(root=raw_data_path + "rawdata", split='train', transform=transform_train)

    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    for data, label in trainloader:
        trainset_data = data
        trainset_label = label

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset_data.cpu().detach().numpy())
    dataset_label.extend(trainset_label.cpu().detach().numpy())

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    data_len = len(dataset_label)

    train_num = int(data_len * 0.75)

    train_dataset_image = dataset_image[:train_num]
    train_dataset_label = dataset_label[:train_num]

    test_dataset_image = dataset_image[train_num:]
    test_dataset_label = dataset_label[train_num:]

    # for training dataset
    train_X, train_y, train_statistic = separate_data((train_dataset_image, train_dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    # for test dataset
    test_X, test_y, test_statistic = separate_data((test_dataset_image, test_dataset_label), num_clients, num_classes,
                                    niid=False, balance=True, partition=None, class_per_client=2)

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
                   niid, balance, partition, 'train')

    # save test dataset and config
    save_each_file(config_path, test_path, test_data, num_clients, num_classes, test_statistic,
                   niid, balance, partition, 'test')

if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # real = True if sys.argv[2] == "realworld" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    niid = True
    balance = False
    partition = 'dir'

    num_clients = 20
    num_classes = 7
    raw_data_path = "fer2013/"

    dataset_name = 'non_iid4robot'

    # generate_fer2013(raw_data_path, dataset_name, num_clients, num_classes, niid, balance, partition)
    generate_fer2013_for_robot(raw_data_path, dataset_name, num_clients, num_classes, niid, balance, partition)