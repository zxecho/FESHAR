import json
import os
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
from plotnine import *


def loadjson(fpt, name):
    if '.json' in name:
        name = name[:-5]
    with open(fpt + '/' + name + '.json', 'r+') as jsonf:
        data = json.load(jsonf)
    return data


# 载入chap2实验数据统计
def load_dataset_stats(dataset, config_name=''):
    if config_name is not None:
        config_name = config_name + '_' + 'config'
    else:
        config_name = 'config'
    data_js = loadjson('../exps/' + dataset, config_name)
    num_clients = data_js['num_clients']
    num_classes = data_js['num_classes']
    stats = data_js['Size of samples for labels in clients']
    for i in range(num_clients):
        print('Client {}'.format(i), stats[i], '\n')

    t_data = np.zeros((num_clients, num_classes))
    for i in range(num_clients):
        for j in range(len(stats[i])):
            t_data[i][stats[i][j][0]] = stats[i][j][1]

    with open('../exps/' + dataset + '/' + 'stats' + '.npz', 'wb') as f:
        np.savez_compressed(f, data=t_data)

    return num_clients


# 绘制构造的数据示意图
def plot_dataset_stats(dataset, mark=None):
    file_path = '../exps/{}'.format(dataset)
    num_clients = load_dataset_stats(dataset, mark)

    stat_np_data = np.load('../exps/' + dataset + '/' + 'stats.npz')
    stats_data = stat_np_data['data']
    # print(stat_np_data['data'])

    stats_df = pd.DataFrame(stats_data).reset_index()
    stats_data = pd.melt(stats_df, id_vars='index', var_name='Classes', value_name='value')
    stats_data['AbsValue'] = np.abs(stats_data.value)
    print('dataframe:', stats_data.head())

    # plotnie 气泡图
    base_plot = (ggplot(stats_data, aes(x='index', y='Classes', fill='value', size='AbsValue')) +
                 geom_point(shape='o', colour="black") +
                 scale_size_area(max_size=12, guide=False) +
                 scale_x_continuous(name='Clients', breaks=np.arange(0, num_clients, 1), limits=(0, num_clients)) +
                 scale_fill_distiller(type='div', palette='RdYlBu', name='Number') +
                 coord_equal() +
                 theme(dpi=100, figure_size=(9, 4))
    )
    print(base_plot)
    # fig, ax = plt.subplots()
    # pcm = plt.pcolormesh(stats_data * (stats_data.shape[1] + 1),
    #                      cmap='viridis')
    # fig.colorbar(pcm, ax=ax)
    #
    # plt.show()


# 载入FedFER数据统计
def load_FedFER_datastats():
    label = {"anger": 0, "disgust": 1, "fear": 2, "happy": 3, "sadness": 4, "surprise": 5}
    dir = 'G:/实验室/实验室项目资料/联邦表情识别/pro_codes/FedFER_pro/dataset/jaffe_Fed_NonIID_L4/fam_stat/'
    file_name = 'fam0_test_stat_info'
    train_dir = dir + 'train'
    test_dir = dir + 'test'
    num_clients = len(os.listdir(train_dir))

    stats_data = np.zeros((num_clients, 6))

    for i in range(num_clients):
        fm_stat = loadjson(train_dir, 'fam{}_{}_stat_info'.format(i, 'train'))
        for k, v in zip(fm_stat.keys(), fm_stat.values()):
            stats_data[i][label[k]] = v

    stats_df = pd.DataFrame(stats_data).reset_index()
    stats_data = pd.melt(stats_df, id_vars='index', var_name='Classes', value_name='value')
    stats_data['AbsValue'] = np.abs(stats_data.value)
    print('dataframe:', stats_data.head())

    # plotnie 气泡图
    base_plot = (ggplot(stats_data, aes(x='index', y='Classes', fill='value', size='AbsValue')) +
                 geom_point(shape='o', colour="black") +
                 scale_size_area(max_size=12, guide=False) +
                 scale_x_continuous(name='Clients', breaks=np.arange(0, num_clients, 1), limits=(0, num_clients)) +
                 scale_fill_distiller(type='div', palette='RdYlBu', name='Number') +
                 coord_equal() +
                 theme(dpi=100, figure_size=(9, 4))
    )
    print(base_plot)


# 用于绘制联邦模型的测试集准确率
def plot_fed_avg_acc(dataset_name='mnist', FL_param='acc', exp_select='E', insert_zone=False):
    """
    用于绘制联邦模型的测试集准确率
    :return:
    """
    files_path = 'G:/实验室/实验室项目资料/家庭陪护机器人相关交互研究/项目和实验/实验结果/{}_exps_show/{}/'.format(dataset_name, exp_select)

    exps = os.listdir(files_path)

    # 创建绘图
    plt.style.use('seaborn-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    # 嵌入绘制局部放大图的坐标系
    axins = inset_axes(ax, width="40%", height="30%", loc='center left',
                       bbox_to_anchor=(0.5, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)

    est_data_list = []

    for exp in exps:

        times = 3
        acc_data_list = []
        auc_data_list = []
        loss_data_list = []

        for i in range(times):
            f = h5py.File(files_path + exp + '/' + '{}_FedAvg_test_{}.h5'.format(dataset_name, i), 'r')
            # keys ['rs_test_acc', 'rs_test_auc', 'rs_train_loss']
            print(f.keys())

            acc = f['rs_test_acc']
            auc = f['rs_test_auc']
            loss = f['rs_train_loss']
            x_axis_index = np.arange(len(acc))

            acc_data_list.append(acc)
            auc_data_list.append(auc)
            loss_data_list.append(loss)

        if FL_param == 'acc':
            data_array = np.vstack(acc_data_list)
            YLabel = 'Test accuracy'
        elif FL_param == 'auc':
            data_array = np.vstack(auc_data_list)
            YLabel = 'AUC'
        else:
            YLabel = 'Loss'
            data_array = np.vstack(loss_data_list)

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, data, label_name='', **kw):
            est = np.mean(data, axis=0)
            sd = np.std(data, axis=0)
            cis = (est - sd, est + sd)

            est_data_list.append(est)

            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=label_name, **kw)
            ax.tick_params(labelsize=15)
            ax.set_ylabel(YLabel, size=15)
            ax.set_xlabel('#Communication rounds', size=15)
            ax.margins(x=0)

            # 在子坐标系中绘制原始数据
            axins.plot(x, est)

            return est, sd

        # xaxis_from_sets = []
        # all_data = np.concatenate(datafile_list, axis=1)
        indicator_avg, indicator_std = tsplot(ax, x_axis_index, data_array,
                                              label_name='{}'.format(exp))  # .split('_')[-1]

    if insert_zone:

        # 设置放大区间
        zone_left = 160
        zone_right = 180

        # 坐标轴的扩展比例（根据实际数据调整）
        x_ratio = 0.1  # x轴显示范围的扩展比例
        y_ratio = 0.1  # y轴显示范围的扩展比例

        # X轴的显示范围
        xlim0 = x_axis_index[zone_left] - (x_axis_index[zone_right] - x_axis_index[zone_left]) * x_ratio
        xlim1 = x_axis_index[zone_right] + (x_axis_index[zone_right] - x_axis_index[zone_left]) * x_ratio

        # Y轴的显示范围
        est_data_array = np.vstack(est_data_list)
        y = est_data_array[:, zone_left:zone_right]
        # y = np.hstack((est_data_list[zone_left:zone_right], y_2[zone_left:zone_right], y_3[zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)

        # 建立父坐标系与子坐标系的连接线
        # loc1 loc2: 坐标系的四个角
        # 1 (右上) 2 (左上) 3(左下) 4(右下)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)      # (3,1) (2,4)

    plt.grid(True)
    ax.legend(loc='upper left', fontsize=12)    # center left / upper / lower
    plt.show()


if __name__ == '__main__':
    dataset = 'fer2013/non_iid(n20nc7d0.1)'
    plot_dataset_stats(dataset, mark=None)
    # load_FedFER_datastats()
    # plot_fed_avg_acc(dataset_name=dataset, FL_param='loss', exp_select='EKB', insert_zone=True)
