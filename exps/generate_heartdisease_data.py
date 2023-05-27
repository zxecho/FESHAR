import numpy as np
import pandas as pd
from sklearn import preprocessing

import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from exps.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)


def get_hd_cleveland_data():
    # data = np.loadtxt('./heart_disease/processed_cleveland.data',  delimiter=',')
    data = pd.read_csv('./heart_disease/processed_cleveland.data')

    data.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                    'slope', 'ca', 'thal', 'num']

    X = data.loc[(data['ca'] != '?') & (data['thal'] != '?')].values[:, :-1]
    y = data.loc[(data['ca'] != '?') & (data['thal'] != '?')].values[:, -1]

    X = X.astype(dtype=np.float)

    print(X)
    print(y)


def generate_hd_data(dir_path, num_clients, num_classes, niid, real, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # if check(config_path, train_path, test_path, num_clients, num_classes, niid, real, partition):
    #     return

    dataset = pd.read_csv('./heart_disease/data.csv')

    dataset.fillna(dataset[1:].mean(), inplace=True)

    dataset_to_array = np.array(dataset)
    label = dataset_to_array[:, 57]  # "Target" classes having 0 and 1
    label = label.astype('int')
    label[label > 0] = 1  # When it is 0 heart is healthy, 1 otherwise

    # extracting 13 features
    dataset = np.column_stack((
        dataset_to_array[:, 4],  # pain location
        dataset_to_array[:, 6],  # relieved after rest
        dataset_to_array[:, 9],  # pain type
        dataset_to_array[:, 11],  # resting blood pressure
        dataset_to_array[:, 33],  # maximum heart rate achieved
        dataset_to_array[:, 34],  # resting heart rate
        dataset_to_array[:, 35],  # peak exercise blood pressure (first of 2 parts)
        dataset_to_array[:, 36],  # peak exercise blood pressure (second of 2 parts)
        dataset_to_array[:, 38],  # resting blood pressure
        dataset_to_array[:, 39],  # exercise induced angina (1 = yes; 0 = no)
        dataset.age,  # age
        dataset.sex,  # sex
        dataset.hypertension  # hyper tension
    ))

    dataset = np.array(dataset, dtype='float')
    # 对数据进行正则化
    dataset = preprocessing.normalize(dataset, axis=1)

    # dataset.astype('float')

    print("The Dataset dimensions are : ", dataset.shape, "\n")

    X, y, statistic = separate_data((dataset, label), num_clients, num_classes,
                                    niid, real, partition, balance=True)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, real, partition)


if __name__ == '__main__':
    num_clients = 5
    num_classes = 2
    dir_path = "heart_disease/"

    niid = True
    real = False
    partition = None

    generate_hd_data(dir_path, num_clients, num_classes, niid, real, partition)

