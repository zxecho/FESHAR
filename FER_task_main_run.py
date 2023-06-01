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
from system_layer.servers.FER_Task.FER_server import FERserver
from system_layer.servers.FER_Task.FedPer_server import FedPer
from system_layer.servers.server_rep import FedRep
from system_layer.servers.server_fedprox import FedProx
from system_layer.servers.server_ditto import Ditto
from system_layer.servers.only_local_train import OnlyLocalTrain_server
# 载入模型
from algo_layer.models.cnn import FedAvgCNN
from algo_layer.models.fcn import FedAvgMLP
from algo_layer.models.lightCNN import MySimpleNet as SimpleNeXt
from algo_layer.models.ConvNeXt_v1 import ConvNeXt
from algo_layer.models.model_utils import LocalModel, BaseHeadSplit
# 配置参数
from system_layer.configs import args_parser
# 载入全局状态监控
from cross_layer.results_monitor import average_data, local_average_results
from cross_layer.process_logging import save_args_config


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
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=5184).to(args.device)
        elif args.model_name == 'mlp':
            args.model = FedAvgMLP(in_features=13, num_classes=args.num_classes, hidden_dim=128).to(args.device)
        elif args.model_name == 'simplenet':
            args.model = SimpleNeXt(classes=6).to(args.device)
        elif args.model_name == 'ConvNeXt_base':
            args.model = ConvNeXt(num_classes=args.num_classes, depths=[3, 3, 27, 3],
                                  dims=[128, 256, 512, 1024]).to(args.device)
        elif args.model_name == 'ConvNeXt_attom':
            args.model = ConvNeXt(num_classes=args.num_classes, depths=[2, 2, 4, 2],
                                  dims=[32, 64, 128, 256]).to(args.device)

        # 选择算法
        if args.algorithm == 'FedAvg':
            server = FERserver(args, i)
        elif args.algorithm == 'only_local':
            server = OnlyLocalTrain_server(args, i)
        elif args.algorithm == 'FedProx':
            server = FedProx(args, i)
        elif args.algorithm == "FedPer":
            args.predictor = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = FedPer(args, i)
        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)
        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        else:
            raise NotImplementedError

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

    # 全局平均


if __name__ == '__main__':
    total_start = time.time()

    args = args_parser()

    with open('./system_layer/configs/ex1_config.yaml', encoding='utf-8') as f:
        data_configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # general config settings
    #   times: 1
    #   model: cnn
    #   predictor: cnn
    #   algorithm: FedAvg
    #   global_rounds: 200
    #   batch_size: 8
    #   local_steps: 5
    general_params = data_configs.pop('general_config')
    args.times = general_params['times']
    args.model_name = general_params['model_name']
    args.predictor = general_params['predictor']
    args.algorithm = general_params['algorithm']
    args.global_rounds = general_params['global_rounds']
    args.optimizer = general_params['optimizer']
    args.batch_size = general_params['batch_size']
    args.local_steps = general_params['local_steps']
    if general_params['local_per_opt']:
        args.local_per_opt = general_params['local_per_opt']
        per_local_setttings = data_configs.pop('local_per_settings')
        args.local_per_optimizer = per_local_setttings['local_per_optimizer']
        args.local_per_lr_scheduler = per_local_setttings['local_per_lr_scheduler']

    run_exps_params = data_configs.keys()
    print(run_exps_params)
    for param in run_exps_params:
        args.num_clients = data_configs[param]['num_clients']
        args.dataset = data_configs[param]['dataset']
        args.save_folder_name = '{}_{}_FER_{}_exp-N3'.format(args.model_name, args.algorithm, param)

        # print essential parameters info
        print("=" * 50)

        print("Algorithm: {}".format(args.algorithm))
        print("Local batch size: {}".format(args.batch_size))
        print("Local steps: {}".format(args.local_steps))
        print("Local learing rate: {}".format(args.local_learning_rate))
        print("Total number of clients: {}".format(args.num_clients))
        print("Clients join in each round: {}".format(args.join_ratio))
        print("Client drop rate: {}".format(args.client_drop_rate))
        print("Time select: {}".format(args.time_select))
        print("Time threthold: {}".format(args.time_threthold))
        print("Global rounds: {}".format(args.global_rounds))
        print("Running times: {}".format(args.times))
        print("Dataset: {}".format(args.dataset))
        print("Local model: {}".format(args.model))
        print("Using device: {}".format(args.device))

        if args.device == "cuda":
            print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        print("=" * 50)

        # save args
        save_args_config(args)

        run(args)

        # 清空GPU缓存
        torch.cuda.empty_cache()
