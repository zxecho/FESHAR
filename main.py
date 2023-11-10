import logging
import os
import time
import copy
import yaml
import numpy as np

import torch
# 服务器server
# 任务服务器模型算法
# facial expression recognition experiments
from system_layer.servers.server_fedavg import FedAvg
from system_layer.servers.server_dyn import FedDyn
from system_layer.servers.FER_Task.FedPer_server import FedPer
from system_layer.servers.server_rep import FedRep
from system_layer.servers.server_rod import FedROD
from system_layer.servers.server_babu import FedBABU
from system_layer.servers.server_fedprox import FedProx
from system_layer.servers.server_ditto import Ditto
from system_layer.servers.server_amp import FedAMP
from system_layer.servers.server_fedgen import FedGen
from system_layer.servers.FER_Task.my_FER_server import FedPLAG
from system_layer.servers.only_local_train import OnlyLocalTrain_server
# 载入模型
from algo_layer.models.cnn import FedAvgCNN
from algo_layer.models.fcn import FedAvgMLP
from algo_layer.models.lightCNN import MySimpleNet as SimpleNeXt
from algo_layer.models.ConvNeXt_v1 import ConvNeXt
from algo_layer.models.vgg import VGG
from algo_layer.models.resnet import ResNet18
from algo_layer.models.model_utils import LocalModel, BaseHeadSplit
# 配置参数
from system_layer.configs import args_parser
# 载入全局状态监控
from cross_layer.results_monitor import average_data, local_average_results
from cross_layer.process_logging import save_args_config

from infrastructure_layer.basic_utils import count_vars_module


def run(args):
    time_list = []
    # reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # 初始化模型
        if args.model_name == 'cnn':
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif args.model_name == 'mlp':
            args.model = FedAvgMLP(in_features=13, num_classes=args.num_classes, hidden_dim=128).to(args.device)
        elif args.model_name == 'simplenet':
            args.model = SimpleNeXt(classes=6, stem_dims=args.stem_channels).to(args.device)
        elif args.model_name == 'simplenet_no_split_stem':
            args.model = SimpleNeXt(classes=6, stem_dims=args.stem_channels).to(args.device)
        elif args.model_name == 'ConvNeXt_base':
            args.model = ConvNeXt(num_classes=args.num_classes, depths=[3, 3, 27, 3],
                                  dims=[128, 256, 512, 1024]).to(args.device)
        elif args.model_name == 'ConvNeXt_attom':
            args.model = ConvNeXt(num_classes=args.num_classes, depths=[2, 2, 4, 2],
                                  dims=[32, 64, 128, 256]).to(args.device)
        elif args.model_name == 'resnet':
            args.model = ResNet18(classes_num=args.num_classes).to(args.device)
        elif args.model_name == 'vgg':
            args.model = VGG(in_chanels=3, class_num=args.num_classes, vgg_name='VGG11')

        # 选择算法
        if args.algorithm == 'FedAvg':
            server = FedAvg(args, i)
        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)
        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)
        elif args.algorithm == 'only_local':
            server = OnlyLocalTrain_server(args, i)
        elif args.algorithm == "FedPLAG":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPLAG(args, i)
        elif args.algorithm == 'FedProx':
            server = FedProx(args, i)
        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = LocalModel(args.model, args.head)
            server = FedPer(args, i)
        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)
        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)
        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)
        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)
        else:
            raise NotImplementedError

        print('>>>>>>>>>>> [Model Parameters: {}] <<<<<<<<<<<<'.format(count_vars_module(args.model)))
        print('>>>>>>>>>>> [Shared Parameters: {}] <<<<<<<<<<<<'.format(count_vars_module(args.model, args.layer_idx)))

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    if args.algorithm != 'only_local':
        # Global average
        average_data(dataset=args.dataset,
                     algorithm=args.algorithm,
                     goal=args.goal,
                     save_dir=args.save_folder_name,
                     times=args.times,
                     length=args.global_rounds / args.eval_gap + 1)
    else:
        local_average_results(server.each_client_max_test_acc, server.each_client_max_test_auc,
                              args.save_folder_name)

    print("All done!")

    # reporter.report()

    # save args again due to some changes in trianing progress
    save_args_config(args)


if __name__ == '__main__':
    total_start = time.time()

    args = args_parser()
    args.save_folder_name = '{}_{}_{}_baseline'.format(args.model_name, args.algorithm, args.dataset)

    # print essential parameters info
    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    print("=" * 50)

    # save args
    save_args_config(args)

    run(args)

    # 清空GPU缓存
    torch.cuda.empty_cache()