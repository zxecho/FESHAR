import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from infrastructure_layer.basic_utils import mkdir

batch_size = 64
train_size = 0.75
least_samples = int(batch_size / (1 - train_size))
alpha = 0.1


def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          real=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['real_world'] == real and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes, niid=False, real=True, partition=None,
                  balance=False, class_per_client=2):
    """
    Split data into two part: training and testing.

    least_samples: the least samples in a class # batch_size / (1 - train_size)
    """
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    if partition is None or partition == "noise":
        # split dataset into classes
        dataset = []
        for i in range(num_classes):
            idx = dataset_label == i
            dataset.append(dataset_content[idx])

        # if niid is False or real is True, set class_per_client to num_classes
        if not niid or real:
            class_per_client = num_classes

        # set class_num_client to the number of classes for each client
        class_num_client = [class_per_client for _ in range(num_clients)]
        # for each class, select clients to be included in the dataset
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_client[client] > 0:
                    selected_clients.append(client)
            # if niid is True and real is False, select clients based on the number of classes
            if niid and not real:
                selected_clients = selected_clients[:int(num_clients / num_classes * class_per_client)]

            # calculate the number of samples for each client
            num_all = len(dataset[i])
            num_clients_ = len(selected_clients)
            # if niid is True and real is True, select the number of clients randomly
            if niid and real:
                num_clients_ = np.random.randint(1, len(selected_clients))
            num_per = num_all / num_clients_
            # if balance is True, set the number of samples for each client to the number of samples for each client
            if balance:
                num_samples = [int(num_per) for _ in range(num_clients_)]
            # if balance is False, set the number of samples for each client randomly
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes),
                                                num_per, num_clients_ - 1).tolist()
            # set the number of samples for each client
            num_samples.append(num_all - sum(num_samples))

            # if niid is True, select clients randomly
            if niid:
                # each client is not sure to have all the labels
                selected_clients = list(np.random.choice(selected_clients, num_clients_, replace=False))

            # assign data to each client
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if len(X[client]) == 0:
                    X[client] = dataset[i][idx:idx + num_sample]
                    y[client] = i * np.ones(num_sample)
                else:
                    X[client] = np.append(X[client], dataset[i][idx:idx + num_sample], axis=0)
                    y[client] = np.append(y[client], i * np.ones(num_sample), axis=0)
                idx += num_sample
                statistic[client].append((i, num_sample))
                class_num_client[client] -= 1

        # if niid and real and partition == "noise":
        #     for client in range(num_clients):
        #         # X[client] = list(map(float, X[client]))
        #         X[client] = np.array(X[client])
        #         X[client] += np.random.normal(0, sigma * client / num_clients)
        #         X[client] = X[client]

    elif niid and partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)
        net_dataidx_map = {}

        # Create a batch of clients
        while min_size <= least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                # Balance
                # 将小于N/Nk的客户端该标签的数据概率设置为1，其余设置为0
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                # 再重新归一化
                proportions = proportions / proportions.sum()
                # 计算对应概率百分比的数量
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # 根据概率划分数据，取出该数据
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # Create a dictionary to store the data and labels of each client
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # additional codes
        # Create a dictionary to store the data and labels of each client
        for client in range(num_clients):
            idxs = net_dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

            # Create a list to store the number of samples of each label
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client] == i))))
    else:
        raise EOFError

    del data
    # gc.collect()

    # Print the number of data and labels of each client
    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def separate_data_v2(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=False)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")
    path_dir_list = train_path.split('/')
    train_path = os.path.join(path_dir_list[0], path_dir_list[1] + '(n{}nc{}d{})'.format(num_clients, num_classes, alpha), path_dir_list[2])
    mkdir(train_path)
    for idx, train_dict in enumerate(train_data):
        with open(train_path + '/' + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    path_dir_list = test_path.split('/')
    test_path = os.path.join(path_dir_list[0], path_dir_list[1] + '(n{}nc{}d{})'.format(num_clients, num_classes, alpha), path_dir_list[2])
    mkdir(test_path)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + '/' + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    # save config file
    path_dir_list = config_path.split('/')
    config_path = os.path.join(path_dir_list[0], path_dir_list[1] + '(n{}nc{}d{})'.format(num_clients, num_classes, alpha)) + "/config.json"
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")


def save_each_file(config_path, data_path, data, num_clients, num_classes, statistic,
                   niid=False, real=True, partition=None, mark='train'):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'real_world': real,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")
    path_dir_list = data_path.split('/')
    train_path = os.path.join(path_dir_list[0],
                              path_dir_list[1] + '(n{}nc{}d{})'.format(num_clients, num_classes, alpha),
                              path_dir_list[2])
    mkdir(train_path)
    for idx, train_dict in enumerate(data):
        with open(train_path + '/' + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)

    # save config file
    path_dir_list = config_path.split('/')
    config_path = os.path.join(path_dir_list[0], path_dir_list[1] + '(n{}nc{}d{})'.format(num_clients, num_classes, alpha)) + f"/{mark}_config.json"
    with open(config_path, 'w') as f:
        ujson.dump(config, f)
