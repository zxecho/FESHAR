import ujson
import numpy as np
from PIL import Image
import os
import torch
import torch.utils.data as data
from infrastructure_layer.read_data import read_data
from infrastructure_layer.read_data import read_client_data_text


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


class FER_Dataset(data.Dataset):
    """
    构建FER的torch数据集格式，原数据集图片尺寸168x168的灰度图片
    """

    def __init__(self, dataset_name, is_train, transform=None):
        self.transform = transform
        # 读取保存的本地数据
        if is_train:
            train_data_dir = os.path.join(dataset_name, 'train/')

            data_file = train_data_dir + 'train.npz'
        else:
            test_data_dir = os.path.join(dataset_name, 'test/')

            data_file = test_data_dir + 'test.npz'

        with open(data_file, 'rb') as f:
            data_ = np.load(f, allow_pickle=True)['data'].tolist()

        self.data_array = data_['x']
        self.label_array = data_['y']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_array[index], self.label_array[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.data_array.shape[0]

