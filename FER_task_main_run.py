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
# 载入模型
from algo_layer.models.cnn import FedAvgCNN
from algo_layer.models.fcn import FedAvgMLP
from algo_layer.models.lightCNN import MySimpleNet as SimpleNeXt
from algo_layer.models.fcn import LocalModel
# 配置参数
from system_layer.configs import args_parser
# 载入全局状态监控
from cross_layer.results_monitor import average_data
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
        if args.model == 'cnn':
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=5184).to(args.device)
        elif args.model == 'mlp':
            args.model = FedAvgMLP(in_features=13, num_classes=args.num_classes, hidden_dim=128).to(args.device)
        elif args.model == 'simplenet':
            args.model = SimpleNeXt(classes=6).to(args.device)
        # 选择算法
        if args.algorithm == 'FedAvg':
            server = FERserver(args, i)
        elif args.algorithm == "FedPer":
            args.predictor = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = LocalModel(args.model, args.predictor)
            server = FedPer(args, i)
        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset,
                 algorithm=args.algorithm,
                 goal=args.goal,
                 save_dir=args.save_folder_name,
                 times=args.times,
                 length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    # reporter.report()

    # 全局平均


if __name__ == '__main__':
    total_start = time.time()

    args = args_parser()

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

    with open('./system_layer/configs/ex1_config.yaml', encoding='utf-8') as f:
        data_configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    run_exps_params = data_configs.keys()
    print(run_exps_params)
    for param in run_exps_params:
        args.num_clients = data_configs[param]['num_clients']
        args.local_steps = data_configs[param]['local_steps']
        args.dataset = data_configs[param]['dataset']
        args.save_folder_name = 'cnn_{}_FER_{}_exp4'.format(args.algorithm, param)

        # save args


        run(args)

        # 清空GPU缓存
        torch.cuda.empty_cache()
