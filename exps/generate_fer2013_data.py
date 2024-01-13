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

    dataset_name = 'non_iid'

    generate_fer2013(raw_data_path, dataset_name, num_clients, num_classes, niid, balance, partition)
