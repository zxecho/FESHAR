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

