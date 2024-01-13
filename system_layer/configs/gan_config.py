import argparse
import logging
import os


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Federated Training')
    parser.add_argument('-log', "--logger", type=object, default=None, help="Log handler")
    # ============ FL parameters =====================
    # 实验仿真参数
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('--dataset', type=str, default='mnist/non_iid4robot(n20nc10d0.1)',
                        help="dataset name(oulu, heart_disease, mnist, cifar10)")
    parser.add_argument('--save_folder_name', type=str, default='FedAvg_FER_oulu_ex3-1', help="save folder name")
    parser.add_argument('--save_folder_path', type=str, default='.', help="save folder path")
    parser.add_argument('--goal', type=str, default="test", help="exps goal")
    parser.add_argument('-algo', "--algorithm", type=str, default="pfedg")
    parser.add_argument('-mn', "--model_name", type=str, default="lenet5", choices=["lenet5", "resnet18", 'mine'])
    parser.add_argument('-m', "--model", type=object, default=None, help="for model")
    parser.add_argument('-dev', "--device", type=str, default="cuda:0", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")

    # 服务器实验参数设置
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.25, help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Total number of clients")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    # 参与客户端参数设置
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-lbs', "--batch_size", type=int, default=8)
    parser.add_argument('-opt', "--optimizer", type=str, default="SGD")
    parser.add_argument('-lfc', "--loss_fc", type=str, default="cross_entropy")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay for optimizer (default: 1e-4)")
    parser.add_argument('-lrs', "--lr_scheduler", type=str, default="Exponential")
    parser.add_argument('-lpo', "--local_per_opt", type=bool, default=False, help='local personalization optim')
    parser.add_argument('-lopt', "--local_per_optimizer", type=str, default="SGD")
    parser.add_argument('-lplrs', "--local_per_lr_scheduler", type=str, default="Exponential")

    parser.add_argument('-glr', "--global_learning_rate", type=float, default=3e-4, help="Global learning rate")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument('-mlr', "--min_lr", type=float, default=1e-6, help="Local minimal learning rate")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.995)
    parser.add_argument('-lde', "--lr_decay_every", type=int, default=10)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    # 系统设置
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-sf', "--save_freq", type=int, default=1, help="Rounds gap for saving")

    # 额外隐私保护策略
    parser.add_argument('-dp', "--privacy", type=bool, default=False, help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)

    # practical模拟实际客户端状况（如延时、丢失、发送延迟等）
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="The rate for slow clients when "
                                                                                   "training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="The rate for slow clients when "
                                                                                  "sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False, help="Whether to group and select clients "
                                                                               "at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="The threthold for droping slow "
                                                                                    "clients")

    # 攻击与防御测试
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlg_method', "--dlg_method", type=str, default='iDLG', choices=['DLG', 'iDLG'])
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=20)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    # ==========================================
    #  具体相关算法参数设置
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg, \
                        or L1 regularization weight of FedTransfer")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0.01,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # FedGen
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # 专门用于联邦的生成网络
    parser.add_argument('--global_iter_per_epoch', type=int, default=100,
                        help="the number of iteration per epoch for server training (default: 100)")
    parser.add_argument('--gan_server_epochs', type=int, default=2000,
                        help="the epochs for server's gan training (default: 20)")
    parser.add_argument('--gan_client_epoch', type=int, default=20,
                        help="the epochs for clients' local gan training (default: 20)")
    parser.add_argument('-gen_lr', "--generator_learning_rate", type=float, default=0.0005)
    parser.add_argument('-tlg', "--train_local_gan", type=bool, default=False)

    # for network input settings
    parser.add_argument('--image_channel', type=int, default=1, help="channel for images")
    parser.add_argument('--image_size', type=int, default=32, help='size for images')
    parser.add_argument('--feature_dim', type=int, default=32, help='feature num for network model')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim for network model')
    parser.add_argument('--feature_size', type=int, default=32, help='feature size for network model')
    parser.add_argument('--input_size', type=tuple, default=(32, 32), help="Input data size")
    parser.add_argument('--input_channels', type=int, default=1, help="Input image channels")
    parser.add_argument('--noise_dim', type=int, default=100,
                        help="the noise dim for the generator input (default: 100)")
    parser.add_argument('--latent_dim', type=int, default=100, help="noise latent dim for generation")
    parser.add_argument('--local_g_optimizer', type=str, default='adam', help="Fed G optimizer")
    parser.add_argument('--local_d_optimizer', type=str, default='adam', help="Fed D optimizer")
    parser.add_argument('--local_g_lr', type=float, default=1e-4, help="Fed G learning rate")
    parser.add_argument('--fed_g_lr_decay', type=float, default=0.9, help="Fed G learning rate decay rate")
    parser.add_argument('--local_d_lr', type=float, default=1e-4, help="Fed D learning rate")
    parser.add_argument('--fed_d_lr_decay', type=float, default=0.9, help="Fed D learning rate decay rate")
    parser.add_argument('--fed_c_lr', type=float, default=2e-3, help="Fed C learning rate")
    parser.add_argument('--fed_c_decay_start', type=int, default=5, help="Fed C learning rate decay start")
    parser.add_argument('--fed_c_lr_decay', type=float, default=0.95, help="Fed C learning rate decay rate")
    parser.add_argument('--fed_c_decay_steps', type=int, default=5, help="Fed C learning rate decay every N steps")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--n_critic', type=int, default=5, help="WGAN n critic")
    parser.add_argument('--clip_value', type=int, default=0.01, help="WGAN discriminator clip value")
    # LLP
    parser.add_argument('-llp', "--llp", type=bool, default=False)
    parser.add_argument('-stemc', "--stem_channels", type=int, default=40)

    # For the FedCG algo
    parser.add_argument('--noise_std', type=float, default=1.,
                        help="std for gaussian noise added to image(default: 1.)")
    parser.add_argument('--add_noise', dest='add_noise', action='store_true', default=False,
                        help="whether adding noise to image")
    parser.add_argument('--distance', type=str, default="none", choices=["none", "mse", "cos"])

    args = parser.parse_args()

    return args


def setup_network_input(args):
    if args.model_name == "lenet5":
        # extractor input size [3*32*32]
        args.image_size = 32
        # extractor ouput size [16*5*5]
        args.feature_dim = 16
        args.feature_size = 4
    elif args.model_name == "alexnet":
        # extractor input size [3*224*224]
        args.image_size = 224
        # extractor ouput size [192*13*13]
        args.feature_dim = 192
        args.feature_size = 13
    elif args.model_name == "resnet18":
        # extractor input size [3*224*224]
        args.image_size = 224
        # extractor ouput size [128*28*28]
        args.feature_dim = 128
        args.feature_size = 28
    if args.model_name == "fedZL_GANets":
        # extractor input size [3*32*32]
        args.image_size = 28
        # extractor ouput size [16*5*5]
        args.feature_dim = 512
        args.feature_size = 4
    if args.model_name == "pFedG":
        # extractor input size [3*32*32]
        args.image_size = 28
        # extractor ouput size [16*5*5]
        args.feature_dim = 512
        args.feature_size = 4



def setup_result_saving_path(args):
    if args.algorithm == "fedcg" or args.algorithm == "fedcg_w":
        args.name = args.algorithm + '_' + args.dataset + str(
            args.num_clients) + '_' + args.model_name + '_' + args.distance  # + '_' + str(args.seed)
    else:
        args.name = args.algorithm + '_' + args.dataset + str(
            args.num_clients) + '_' + args.model_name  # + '_' + str(args.seed)
    args.dir = './results/' + args.save_folder_name + '/' + 'bs' + str(args.batch_size) + 'lr' + str(
        args.local_learning_rate) + 'wd' + str(args.weight_decay)
    args.checkpoint_dir = os.path.join(args.dir, args.name, 'checkpoint')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
