import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Federated Training')
    parser.add_argument('-log', "--logger", type=object, default=None, help="Log handler")
    # ============ FL parameters =====================
    # 实验仿真参数
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('--dataset', type=str, default='mnist/non-iid_dir_ex(dir=0.1)',
                        help="dataset name(oulu, heart_disease, mnist, cifar10)")
    parser.add_argument('--save_folder_name', type=str, default='FedAvg_FER_oulu_ex3-1', help="save folder name")
    parser.add_argument('--save_folder_path', type=str, default='.', help="save folder path")
    parser.add_argument('--goal', type=str, default="test", help="exps goal")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-mn', "--model_name", type=str, default="cnn")
    parser.add_argument('-m', "--model", type=object, default=None, help="for model")
    parser.add_argument('-head', "--head", type=str, default="cnn")
    parser.add_argument('-dev', "--device", type=str, default="cuda:0", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")

    # 服务器实验参数设置
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.5, help="Ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Total number of clients")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-gr', "--global_rounds", type=int, default=50)
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    # 参与客户端参数设置
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-lbs', "--batch_size", type=int, default=8)
    parser.add_argument('-opt', "--optimizer", type=str, default="SGD")
    parser.add_argument('-lfc', "--loss_fc", type=str, default="cross_entropy")
    parser.add_argument('-lrs', "--lr_scheduler", type=str, default="Exponential")
    parser.add_argument('-lpo', "--local_per_opt", type=bool, default=False, help='local personalization optim')
    parser.add_argument('-lopt', "--local_per_optimizer", type=str, default="SGD")
    parser.add_argument('-lplrs', "--local_per_lr_scheduler", type=str, default="Exponential")

    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.0005, help="Local learning rate")
    parser.add_argument('-mlr', "--min_lr", type=float, default=1e-6, help="Local minimal learning rate")
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.995)
    parser.add_argument('-lde', "--lr_decay_every", type=int, default=10)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    # 系统设置
    parser.add_argument('-pv', "--prev", type=int, default=0, help="Previous Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
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
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # 攻击与防御测试
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=20)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
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
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    # FedBABU
    parser.add_argument('-fts', "--fine_tuning_steps", type=int, default=5)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # LLP
    parser.add_argument('-llp', "--llp", type=bool, default=True)
    parser.add_argument('-stemc', "--stem_channels", type=int, default=40)

    args = parser.parse_args()
    return args
