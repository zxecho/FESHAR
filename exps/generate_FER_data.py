import numpy as np
import os
import sys
import random
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from exps.dataset_utils import check, separate_data, split_data, save_file
from infrastructure_layer.read_client_data import FER_Dataset

random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 6
dir_path = "FER/BU3DFE/"

resize = 48
cut_size = 44


# Allocate data to users
def generate_FER_data(dir_path, num_clients, num_classes, niid=False, real=True, partition=None):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, real, partition):
    #     return

    # Get MNIST data
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(resize),
        transforms.Grayscale(3),
        transforms.RandomCrop(cut_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # BU3DFE mean=[0.27676, 0.27676, 0.27676], std=[0.26701, 0.26701, 0.26701]
        # jaffe mean=[0.43192, 0.43192, 0.43192], std=[0.27979, 0.27979, 0.27979]
        # oulu mean=[0.36418, 0.36418, 0.36418], std=[0.20384, 0.20384, 0.20384]
        # ck-48  mean=[0.51194, 0.51194, 0.51194], std=[0.28913, 0.28913, 0.28913]
        # transforms.Normalize(mean=args.dataset_mean_std[args.dataset_name]['mean'],
        #                      std=args.dataset_mean_std[args.dataset_name]['std']),
    ])

    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(resize),
        # transforms.ToTensor(),
        transforms.Grayscale(3),
        transforms.TenCrop(cut_size),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Lambda(
        #     lambda crops: torch.stack([transforms.Normalize(mean=args.dataset_mean_std[args.dataset_name]['mean'],
        #                                                     std=args.dataset_mean_std[args.dataset_name]['std'])(
        #         transforms.ToTensor()(crop)) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    transform = transforms.Compose(
        [transforms.Resize(resize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = FER_Dataset(dir_path, is_train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=False)
    testset = FER_Dataset(dir_path, is_train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=len(testset), shuffle=True)

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


if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # real = True if sys.argv[2] == "realworld" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    niid = True
    real = False
    partition = 'dir'

    generate_FER_data(dir_path, num_clients, num_classes, niid, real, partition)
