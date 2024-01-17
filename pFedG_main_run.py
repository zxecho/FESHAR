import os
import time
import yaml
import copy
import numpy as np

import torch
# 服务器server
# 任务服务器模型算法
# facial expression recognition experiments
from system_layer.servers.GAN_Task.FedCG_server import FedCG
from system_layer.servers.GAN_Task.pFedG_server import pFedG_server
from system_layer.servers.server_fedavg import FedAvg
from system_layer.servers.server_fedprox import FedProx
from system_layer.servers.server_dyn import FedDyn
from system_layer.servers.server_scaffold import SCAFFOLD
from system_layer.servers.server_fedgen import FedGen
from system_layer.servers.server_peravg import PerAvg
from system_layer.servers.server_amp  import FedAMP
from system_layer.servers.server_APFL import APFL
from system_layer.servers.server_apple import APPLE
from system_layer.servers.server_per import FedPer
from system_layer.servers.server_rep import FedRep
from system_layer.servers.server_rod import FedROD
from system_layer.servers.server_babu import FedBABU
from system_layer.servers.server_ditto import Ditto
from system_layer.servers.server_fedgh import FedGH
from system_layer.servers.server_gpfl import GPFL
from system_layer.servers.server_distill import FedDistill
from system_layer.servers.server_proto import FedProto
from system_layer.servers.server_local import Local
# 载入模型
from algo_layer.models.cnn import FedAvgCNN, HARCNN
from algo_layer.models.model_utils import BaseHeadSplit
# 配置参数
from system_layer.configs.gan_config import args_parser, setup_result_saving_path, setup_network_input
# 载入全局状态监控
from cross_layer.results_monitor import average_data, local_average_results
from cross_layer.process_logging import save_args_config
# 获取记录日志模块
from cross_layer.process_logging import get_logger


def run(args):
    time_list = []
    # reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # 初始化模型
        # Initialize generator and discriminator
        if args.model_name == 'cnn':
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "fer" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1600).to(args.device)
        elif args.model_name == "harcnn":
            if 'har' in args.dataset:
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes,
                                    conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            elif 'pamap' in args.dataset:
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes,
                                    conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
        elif args.model_name == 'lenet5':
            from algo_layer.models.GAN_modules import Generator
            from algo_layer.models.GAN_modules import Conditional_D as Discriminator
            args.G_model = Generator(n_classes=args.num_classes,
                                     latent_dim=args.noise_dim,
                                     feature_num=args.feature_dim)
            args.D_model = Discriminator(n_classes=args.num_classes,
                                         feature_num=args.feature_dim,
                                         feature_size=args.feature_size)
        elif args.model_name == 'pFedG':
            from algo_layer.models.GAN_models.fedZL_GANets import (Generator, Discriminator,
                                                                   Extractor, Classifier, Generative_model)
            args.model = torch.nn.ModuleDict()
            # args.model['generator'] = Generator(num_classes=args.num_classes,
            #                                     feature_num=args.feature_dim,
            #                                     noise_dim=args.noise_dim)
            args.model['generator'] = Generative_model(noise_dim=args.noise_dim, num_classes=args.num_classes,
                                                       hidden_dim=args.hidden_dim, feature_dim=args.feature_dim,
                                                       device=args.device)
            args.model['discriminator'] = Discriminator(num_classes=args.num_classes,
                                                        feature_size=args.feature_size,
                                                        feature_num=args.feature_dim)

            if "mnist" in args.dataset:
                args.model['classifier'] = Classifier(num_classes=args.num_classes)
                args.model['extractor'] = Extractor(image_channel=args.image_channel, feature_in_dim=1024,
                                                    feature_out_dim=args.feature_dim)
            elif "cifar10" in args.dataset:
                args.model['classifier'] = Classifier(num_classes=args.num_classes)
                args.model['extractor'] = Extractor(image_channel=args.input_channels, feature_in_dim=1600,
                                                    feature_out_dim=args.feature_dim)
            elif "har" in args.dataset:
                from algo_layer.models.GAN_models.HAR_cnn import Extractor, Classifier
                args.model['classifier'] = Classifier(dim_hidden=1664, num_classes=args.num_classes)
                args.model['extractor'] = Extractor(in_channels=args.input_channels,
                                                    conv_kernel_size=(1, 9), pool_kernel_size=(1, 2))

        # 选择算法
        if args.algorithm == 'FedAvg':
            server = FedAvg(args, i)
        elif args.algorithm == "FedProx":
            server = FedProx(args, i)
        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)
        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)
        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)
        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)
        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)
        elif args.algorithm == "APFL":
            server = APFL(args, i)
        elif args.algorithm == "APPLE":
            server = APPLE(args, i)
        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
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
        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)
        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)
        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)
        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.head)
            args.model.head = torch.nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)
        elif args.algorithm == 'fedcg':
            server = FedCG(args, i)
        elif args.algorithm == 'only_local':
            server = Local(args, i)
        elif args.algorithm == 'pfedg':
            server = pFedG_server(args, i)
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

    with open('system_layer/yaml_configs/pFedG_ex_config.yaml', encoding='utf-8') as f:
        data_configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    # general config settings
    general_params = data_configs.pop('general_config')
    args.times = general_params['times']
    args.algorithm = general_params['algorithm']
    args.global_rounds = general_params['global_rounds']
    args.model_name = general_params['model_name']
    args.join_ratio = general_params['join_ratio']
    args.optimizer = general_params['optimizer']
    args.local_learning_rate = general_params['local_learning_rate']
    args.learning_rate_decay = general_params['learning_rate_decay']
    args.loss_fc = general_params['loss_fc']
    args.batch_size = general_params['batch_size']
    args.local_steps = general_params['local_steps']
    args.train_local_gan = general_params['train_local_gan']
    # if general_params['local_per_opt']:
    #     args.local_per_opt = general_params['local_per_opt']
    #     per_local_setttings = data_configs.pop('local_per_settings')
    #     args.local_per_optimizer = per_local_setttings['local_per_optimizer']
    #     args.local_per_lr_scheduler = per_local_setttings['local_per_lr_scheduler']
    # else:
    #     per_local_setttings = data_configs.pop('local_per_settings')

    run_exps_params = data_configs.keys()
    print(run_exps_params)
    for param in run_exps_params:
        args.num_clients = data_configs[param]['num_clients']
        args.num_classes = data_configs[param]['num_classes']
        args.input_size = data_configs[param]['input_size']
        args.input_channels = data_configs[param]['input_channels']
        args.dataset = data_configs[param]['dataset']
        args.save_folder_name = '{}_{}_{}_test'.format(args.model_name, args.algorithm, param)

        # initial network input
        setup_network_input(args)
        setup_result_saving_path(args)
        # get logger
        args.logger = get_logger(args.dir)
        args.logger.info("#" * 100)
        args.logger.info(str(args))
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
