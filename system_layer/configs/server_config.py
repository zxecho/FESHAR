import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Federated Training')
    # ============ FL parameters
    parser.add_argument('--dataset', type=str, default='cifar10', help="dataset name")
    parser.add_argument('--save_folder_name', type=str, default='exp_cifar10', help="save folder name")
    parser.add_argument('--num_classes', type=int, default=10, help="the number of classes")

    # ============== local model parms ===============
    parser.add_argument('--local_steps', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=16, help="local batch size: B")
    parser.add_argument('--local_lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--local_lr_decay', type=float, default=0.8, help="D learning rate decay rate")

    # privacy protection settings
    parser.add_argument('--privacy', type=bool, default=False, help="If add privacy protection")
    parser.add_argument('--dp_sigma', type=float, default=0.1, help="dp sigma")

    # 专门用于联邦的生成网络
    parser.add_argument('--fed_g_lr', type=float, default=1e-3, help="Fed G learning rate")
    parser.add_argument('--fed_g_lr_decay', type=float, default=0.9, help="Fed G learning rate decay rate")
    parser.add_argument('--fed_d_lr', type=float, default=1e-3, help="Fed D learning rate")
    parser.add_argument('--fed_d_lr_decay', type=float, default=0.9, help="Fed D learning rate decay rate")
    parser.add_argument('--fed_c_lr', type=float, default=2e-3, help="Fed C learning rate")
    parser.add_argument('--fed_c_decay_start', type=int, default=5, help="Fed C learning rate decay start")
    parser.add_argument('--fed_c_lr_decay', type=float, default=0.95, help="Fed C learning rate decay rate")
    parser.add_argument('--fed_c_decay_steps', type=int, default=5, help="Fed C learning rate decay every N steps")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--n_critic', type=int, default=5, help="WGAN n critic")
    parser.add_argument('--clip_value', type=int, default=0.01, help="WGAN discriminator clip value")

    args = parser.parse_args()
    return args