import pandas as pd
import json
import os


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False

# 绘制损失图
def loss_plot(axs, loss_data, name=None):
    axs.plot(range(len(loss_data)), loss_data, label=name)
    axs.tick_params(labelsize=13)
    axs.set_ylabel(name, size=13)
    axs.set_xlabel('Communication rounds', size=13)
    axs.legend()
    # axs.set_title(name)


# ================ 持久化数据 ======================
def save2csv(fpt, data, columns, index):
    data = {key: value for key, value in zip(columns, data)}
    print('*** data: \n', data)
    dataframe = pd.DataFrame(data, columns=columns, index=index)
    # 转置
    dataframe = pd.DataFrame(dataframe.values.T, index=dataframe.columns, columns=dataframe.index)
    dataframe.to_csv(fpt, index=True, sep=',')


def save2json(fpt, data, name):
    mkdir(fpt)
    with open(fpt + '/' + name + '.json', 'w+') as jsonf:
        json.dump(data, jsonf)


def save2pkl(fpt, data, name):
    with open(fpt + '/' + name + '.pkl', 'wb') as pkf:
        pickle.dump(data, pkf)


def load_pkl_file(fpt):
    pkl_file = open(fpt, 'rb')
    data = pickle.load(pkl_file)

    return data