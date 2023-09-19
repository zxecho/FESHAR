import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import h5py
import copy
import time
import random
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from system_layer.servers.server_base import Server
from infrastructure_layer.read_client_data import read_client_data
from infrastructure_layer.plot_data import plot_resutls

from cross_layer.security_monitor import DLG


class GAN_server(Server):

    def __init__(self, args, times):
        super().__init__(args, times)
        # old version
        # self.global_G_model = copy.deepcopy(args.G_model)
        # self.global_D_model = copy.deepcopy(args.D_model)
        # self.global_model = (self.global_G_model, self.global_D_model)
        self.logger = args.logger
        self.global_model = nn.ModuleDict()
        self.global_model["generator"] = copy.deepcopy(args.G_model)
        self.global_model["discriminator"] = copy.deepcopy(args.D_model)

        self.latent_dim = args.latent_dim
        self.save_results_path = None

    def send_models(self, modules=None):
        if modules is None:
            modules = ['generator', 'discriminator']
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model, modules)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self, modules=None):
        if modules is None:
            modules = ['generator', 'discriminator']
        assert (len(self.uploaded_models) > 0)
        assert isinstance(self.uploaded_models[0], tuple)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        self.global_G_model = copy.deepcopy(self.uploaded_models[0]['generator'])
        self.global_D_model = copy.deepcopy(self.uploaded_models[0]['discriminator'])
        for module in modules:
            self.global_model[module] = copy.deepcopy(self.uploaded_models[0][module])

            # set the initial model weights to zero
            for param in self.global_model[module].parameters():
                param.data.zero_()

            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model[module], module)

    def add_parameters(self, w, client_model, module):
        # if module_name == 'G':
        #     for server_param, client_param in zip(self.global_G_model.parameters(), client_model.parameters()):
        #         server_param.data += client_param.data.clone() * w
        # elif module_name == 'D':
        #     for server_param, client_param in zip(self.global_D_model.parameters(), client_model.parameters()):
        #         server_param.data += client_param.data.clone() * w

        for server_param, client_param in zip(self.global_model[module].parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def train_metrics(self):
        pass

    def test_metrics(self):
        pass

    def evaluate(self, acc=None, auc=None, loss=None):
        self.sample_image(n_row=10)

    def get_save_results_dir(self, **kwargs):
        client_id = None
        if 'client_id' in kwargs.keys():
            client_id = kwargs['client_id']
        dataset_dir = self.dataset.split('/')
        if len(dataset_dir) > 1:
            dataset = dataset_dir[-1]
        algo = dataset + "_" + self.algorithm
        if client_id is not None:
            result_path = "./results/{}/client_{}/".format(self.save_folder_name, client_id)
        else:
            result_path = "./results/{}/".format(self.save_folder_name)

        self.save_results_path = result_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)

    def sample_image(self, n_row):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        self.get_save_results_dir()
        # Sample noise
        z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim)), dtype=torch.float, device=self.args.device)
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = torch.tensor(labels, dtype=torch.long, device=self.args.device)
        gen_imgs = self.global_G_model(z, labels)
        save_image(gen_imgs.data, self.save_results_path + "/sampling_images.png", nrow=n_row, normalize=True)
