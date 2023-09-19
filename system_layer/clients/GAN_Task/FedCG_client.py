import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from system_layer.clients.GAN_Task.GAN_client_base import GAN_client
from algo_layer.models.GAN_modules import Extractor, Classifier
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params
from system_layer.training_utils import AvgMeter, add_gaussian_noise


class FedCG_client(GAN_client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.model["extractor"] = Extractor(input_channels=args.image_channel)  # E
        self.model["classifier"] = Classifier(n_classes=self.num_classes)  # C

        self.frozen_net(modules=["extractor", "classifier", "generator", "discriminator"], frozen=True)
        self.EC_optimizer = optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.local_learning_rate,
                                       weight_decay=args.weight_decay)
        self.D_optimizer = optim.Adam(self.get_params(["discriminator"]), lr=args.local_learning_rate,
                                      betas=(0.5, 0.999))
        self.G_optimizer = optim.Adam(self.get_params(["generator"]), lr=args.local_learning_rate, betas=(0.5, 0.999))

        self.model.to(self.device)

        self.BCE_criterion = nn.BCELoss().to(self.device)
        self.CE_criterion = nn.CrossEntropyLoss().to(self.device)
        self.MSE_criterion = nn.MSELoss().to(self.device)
        self.COS_criterion = nn.CosineSimilarity().to(self.device)

        if self.args.add_noise:
            self.noise_std = args.noise_std * id / (self.args.num_clients - 1)
            self.logger.info("client:%2d, noise_std:%2.6f" % (id, self.noise_std))
        else:
            self.logger.info("client:%2d" % (id))

    def get_params(self, modules):
        params = []
        for module in modules:
            params.append({"params": self.model[module].parameters()})
        return params

    def frozen_net(self, modules, frozen):
        """
        frozen the net when need to test
        :param modules: 那些模块需要被冻结
        :param frozen:  是否被冻结
        :return:   None
        """
        for module in modules:
            for param in self.model[module].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.model[module].eval()
            else:
                self.model[module].train()

    def train(self, current_round):
        trainloader = self.load_train_data()
        start_time = time.time()
        # for logging info
        self.logger.info("Training Client %2d's EC Network Start!" % self.id)
        EC_loss_meter = AvgMeter()
        EG_distance_meter = AvgMeter()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # local training progress
        for step in range(max_local_steps):
            # resset the meter
            EC_loss_meter.reset()
            EG_distance_meter.reset()
            # frozen the not shared network parameters
            self.frozen_net(["extractor", "classifier"], False)
            for i, (x, y) in enumerate(trainloader):
                if self.args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                z = torch.randn(x.size(0), self.args.noise_dim, 1, 1).to(self.device)
                # slow the training speed
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # compute the lcoal training loss
                self.EC_optimizer.zero_grad()

                E = self.model["extractor"](x)
                EC = self.model["classifier"](E)
                EC_loss = self.CE_criterion(EC, y)

                G = self.model["generator"](z, y).detach()

                if self.args.distance == "mse":
                    EG_distance = self.MSE_criterion(E, G)
                elif self.args.distance == "cos":
                    EG_distance = 1 - self.COS_criterion(E, G).mean()
                elif self.args.distance == "none":
                    EG_distance = 0
                p = min(current_round / 50, 1.)
                gamma = 2 / (1 + np.exp(-10 * p)) - 1
                EG_distance = gamma * EG_distance

                (EC_loss + EG_distance).backward()
                self.EC_optimizer.step()
                EC_loss_meter.update(EC_loss.item())
                EG_distance_meter.update(EG_distance.item())

            self.frozen_net(["extractor", "classifier"], True)

            EC_loss = EC_loss_meter.get()
            EG_distance = EG_distance_meter.get()
            EC_acc = self.local_test()
            self.logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EG_distance:%2.6f, EC_acc:%2.6f" % (
                self.id, current_round, EC_loss, EG_distance, EC_acc))

        # ================ GAN training progress =====================

        self.logger.info("Training Client %2d's DG Network Start!" % self.id)
        ED_loss_meter = AvgMeter()
        GD_loss_meter = AvgMeter()
        G_loss_meter = AvgMeter()

        for epoch in range(self.args.gan_epoch):
            ED_loss_meter.reset()
            GD_loss_meter.reset()
            G_loss_meter.reset()

            self.frozen_net(["generator", "discriminator"], False)

            for i, (x, y) in enumerate(trainloader):
                if self.args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                z = torch.randn(x.size(0), self.args.noise_dim, 1, 1).to(self.device)
                ones = torch.ones((x.size(0), 1)).to(self.device)
                zeros = torch.zeros((x.size(0), 1)).to(self.device)

                # train discriminator
                def train_discriminator(x, y, z):
                    self.D_optimizer.zero_grad()

                    E = self.model["extractor"](x)
                    ED = self.model["discriminator"](E.detach(), y)
                    ED_loss = self.BCE_criterion(ED, ones)

                    G = self.model["generator"](z, y)
                    GD = self.model["discriminator"](G.detach(), y)
                    GD_loss = self.BCE_criterion(GD, zeros)

                    D_loss = ED_loss + GD_loss
                    D_loss.backward()
                    self.D_optimizer.step()
                    ED_loss_meter.update(ED_loss.item())
                    GD_loss_meter.update(GD_loss.item())

                # train generator with diversity
                def train_generator_with_diversity(y, z):
                    self.G_optimizer.zero_grad()

                    # gen loss
                    G = self.model["generator"](z, y)
                    GD = self.model["discriminator"](G, y)
                    G_loss = self.BCE_criterion(GD, ones)

                    G_loss.backward()
                    self.G_optimizer.step()
                    G_loss_meter.update(G_loss.item())

                train_discriminator(x, y, z)
                train_generator_with_diversity(y, z)

            self.frozen_net(["generator", "discriminator"], True)

            ED_loss = ED_loss_meter.get()
            GD_loss = GD_loss_meter.get()
            G_loss = G_loss_meter.get()
            GC_acc = self.compute_gan_acc()
            self.logger.info("Client:[%2d], Epoch:[%2d], ED_loss:%2.6f, GD_loss:%2.6f, G_loss:%2.6f, GC_acc:%2.6f"
                        % (self.id, epoch, ED_loss, GD_loss, G_loss, GC_acc))

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

    def compute_gan_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch in range(100):
                y = torch.randint(0, self.num_classes, (self.batch_size,)).to(self.device)
                z = torch.randn(self.batch_size, self.noise_dim, 1, 1).to(self.device)
                feat = self.model["generator"](z, y)
                pred = self.model["classifier"](feat)
                correct += torch.sum(torch.tensor((torch.argmax(pred, dim=1) == y), dtype=torch.float))
                total += self.batch_size
        return (correct / total).item()

    def local_test(self):
        testloader = self.load_test_data()
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(testloader):
                if self.args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(self.device)
                y = y.to(self.device)
                feat = self.model["extractor"](x)
                pred = self.model["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()

    '''
    def local_val(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.valloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argrmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()
    '''
