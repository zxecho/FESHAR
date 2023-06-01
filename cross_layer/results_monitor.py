import h5py
import numpy as np
import os
from infrastructure_layer.basic_utils import save2json


def average_data(algorithm="", dataset="", goal="", save_dir="", times=10, length=800):
    dataset_dir = dataset.split('/')
    if len(dataset_dir) > 1:
        dataset = dataset_dir[-1]
    test_acc = get_all_results_for_one_algo(
        algorithm, dataset, goal, save_dir, times, int(length))
    test_acc_data = np.average(test_acc, axis=0)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())
    std = np.std(max_accurancy)
    mean = np.mean(max_accurancy)
    # save to local
    save2json(os.path.join('results/', save_dir), {'std': std, 'mean': mean}, 'results')
    print("std for best accurancy:", std)
    print("mean for best accurancy:", mean)


def local_average_results(all_rs_test_max_acc, all_rs_test_max_auc, save_dir):
    std = np.std(all_rs_test_max_acc)
    mean = np.mean(all_rs_test_max_acc)
    # save to local
    save2json(os.path.join('results/', save_dir), {'std': std, 'mean': mean}, 'results')
    print("std for best accurancy:", std)
    print("mean for best accurancy:", mean)


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", save_dir="", times=10, length=800):
    test_acc = np.zeros((times, length))
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + \
                    algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc[i, :] = np.array(
            read_data_then_delete(file_name, save_dir, delete=False))[:length]

    return test_acc


def read_data_then_delete(file_name, save_dir="", delete=False):
    file_path = "results/" + save_dir + '/' + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


# log training results
def log_training_results():
    pass