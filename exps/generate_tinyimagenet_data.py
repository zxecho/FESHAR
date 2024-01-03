import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from exps.dataset_utils import check, separate_data, split_data, save_file, save_each_file
from torchvision.datasets import ImageFolder, DatasetFolder

random.seed(1)
np.random.seed(1)

# http://cs231n.stanford.edu/tiny-imagenet-200.zip
# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        super(ImageFolder_custom, self).__init__(root=root, transform=transform, target_transform=target_transform)
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# Allocate data to users
def generate_dataset(raw_data_path, dataset_name, num_clients, num_classes, niid, balance, partition):
    dir_path = raw_data_path + dataset_name + "/"
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = ImageFolder_custom(root=raw_data_path + 'rawdata/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolder_custom(root=raw_data_path + 'rawdata/tiny-imagenet-200/val/', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

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

    X, y, statistic = separate_data_v2((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


def generate_dataset4robot(raw_data_path, dataset_name, num_classes, niid, balance, partition):
    dir_path = raw_data_path + dataset_name + "/"
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = ImageFolder_custom(root=raw_data_path + 'rawdata/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolder_custom(root=raw_data_path + 'rawdata/tiny-imagenet-200/val/', transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

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
    niid = True
    balance = True
    partition = 'dir'

    num_clients = 20
    num_classes = 200
    dir_path = "tiny_imagenet/"

    generate_dataset(dir_path, num_clients, num_classes, niid, balance, partition)