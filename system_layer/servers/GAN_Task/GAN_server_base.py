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

from cross_layer.security_monitor import DLG, iDLG


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

        self.global_model.to(self.device)

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

    def receive_models(self, modules=None):
        if modules is None:
            modules = ['generator', 'discriminator']
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

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

    def aggregate_parameters_old(self, modules=None):
        if modules is None:
            modules = ['generator', 'discriminator']
        assert (len(self.uploaded_models) > 0)

        # self.global_model = copy.deepcopy(self.uploaded_models[0])
        for module in modules:
            self.global_model[module] = copy.deepcopy(self.uploaded_models[0][module])

            # set the initial model weights to zero
            for param in self.global_model[module].parameters():
                param.data.zero_()

            for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
                self.add_parameters(w, client_model[module], module)

    def aggregate_parameters(self, modules=None):
        if modules is None:
            modules = ['generator', 'discriminator']
        assert (len(self.uploaded_models) > 0)

        for module in modules:
            avg_param = {}
            weights = []
            params = []
            # 遍历每一个客户端
            for w, client in zip(self.uploaded_weights, self.uploaded_models):
                # 将客户端的网络参数加载到avg_param中
                weights.append(w)
                params.append(client[module].state_dict())

            # 计算每一个客户端的平均参数
            for key in params[0].keys():
                avg_param[key] = params[0][key] * weights[0]
                # 遍历每一个客户端
                for idx in range(1, len(self.uploaded_models)):
                    avg_param[key] += params[idx][key] * weights[idx]
            # 将平均参数加载到global_net中
            self.global_model[module].load_state_dict(avg_param)

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
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def evaluate(self, acc=None, auc=None, loss=None):
        stats_test = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats_test[2]) * 1.0 / sum(stats_test[1])
        test_auc = sum(stats_test[3]) * 1.0 / sum(stats_test[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats_test[2], stats_test[1])]
        aucs = [a / n for a, n in zip(stats_test[3], stats_test[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if auc is None:
            self.rs_test_auc.append(test_auc)
        else:
            acc.append(test_auc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

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
        # z = torch.tensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim)), dtype=torch.float, device=self.device)
        z = torch.randn(n_row, self.args.noise_dim, 1, 1).to(self.device)
        # Get labels ranging from 0 to n_classes for n rows
        # labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = np.array([num for num in range(n_row)])
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        gen_imgs = self.global_model["generator"](z, labels)
        save_image(gen_imgs.data, self.save_results_path + "/sampling_images.png", nrow=n_row, normalize=True)

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model = client_model['discriminator']
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model['discriminator'].parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    z = torch.randn(x.size(0), self.args.noise_dim, 1, 1).to(self.device)
                    gen_imgs = self.global_model["generator"](z, y)
                    output = client_model(gen_imgs)
                    target_inputs.append((x, output))
            if self.dlg_method == 'DLG':
                d = DLG(client_model, origin_grad, target_inputs)
            elif self.dlg_method == 'iDLG':
                d = iDLG(client_model, origin_grad, target_inputs)

            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))
        psnr = psnr_val / cnt
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr))
        else:
            print('PSNR error')

        self.rs_test_dlg.append(psnr)
        # self.save_item(items, f'DLG_{R}')