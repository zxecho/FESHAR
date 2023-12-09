import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from system_layer.clients.client_base import Client
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params
from algo_layer.optimizers import get_optimizer
from algo_layer.loss_function_factors import get_loss_function
from algo_layer.lr_scheduler import get_lr_scheduler
from infrastructure_layer.read_client_data import read_client_data


class GAN_client(Client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.logger = args.logger

        self.model = nn.ModuleDict()

        # self.model["generator"] = copy.deepcopy(args.G_model)  # G
        # self.model["discriminator"] = copy.deepcopy(args.D_model)  # D
        self.model["generator"] = copy.deepcopy(args.model['generator'])   # E
        self.model["discriminator"] = copy.deepcopy(args.model['discriminator'])   # C

        self.latent_dim = args.latent_dim
        self.noise_dim = args.noise_dim

        self.g_learning_rate = args.local_g_lr
        self.d_learning_rate = args.local_d_lr

        self.set_optimization()

    def set_optimization(self):
        self.loss = get_loss_function(self.args.loss_fc)
        self.g_optimizer = get_optimizer(self.args.local_g_optimizer, self.model['generator'].parameters(), self.g_learning_rate)
        self.d_optimizer = get_optimizer(self.args.local_d_optimizer, self.model['discriminator'].parameters(), self.d_learning_rate)

    def set_parameters(self, global_model, modules):
        # models is the global model, self.model is the local model
        # for l_model, g_model in zip(self.model, models):
        #     for new_param, old_param in zip(g_model.parameters(), l_model.parameters()):
        #         old_param.data = new_param.data.clone()

        for module in modules:
            global_param = global_model[module].state_dict()
            self.model[module].load_state_dict(global_param)

    def train_metrics(self):
        pass

    def test_metrics(self):
        pass

