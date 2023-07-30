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

        self.model = (copy.deepcopy(args.G_model), copy.deepcopy(args.D_model))
        self.latent_dim = args.latent_dim

        self.g_model = self.model[0]
        self.d_model = self.model[1]

        self.g_learning_rate = args.local_g_lr
        self.d_learning_rate = args.local_d_lr

        self.set_optimization()

    def set_optimization(self):
        self.loss = get_loss_function(self.args.loss_fc)
        self.g_optimizer = get_optimizer(self.args.local_g_optimizer, self.model[0].parameters(), self.g_learning_rate)
        self.d_optimizer = get_optimizer(self.args.local_d_optimizer, self.model[1].parameters(), self.d_learning_rate)

        # self.lr_scheduler = get_lr_scheduler(self.args.lr_scheduler, self.optimizer,
        #                                                 gamma=self.args.learning_rate_decay_gamma)

    def set_parameters(self, models):
        # models is the global model, self.model is the local model
        for l_model, g_model in zip(self.model, models):
            for new_param, old_param in zip(g_model.parameters(), l_model.parameters()):
                old_param.data = new_param.data.clone()

    def train_metrics(self):
        pass

    def test_metrics(self):
        pass

