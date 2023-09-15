import copy
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from system_layer.clients.GAN_Task.GAN_client_base import GAN_client
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params

from training_utils import frozen_net, AvgMeter


class ACGAN_Clinet(GAN_client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.net = nn.ModuleDict()

        self.net["extractor"] = Extractor()  # E
        self.net["classifier"] = Classifier()  # C
        self.net["generator"] = Generator()  # D
        self.net["discriminator"] = Discriminator()  # G
        self.frozen_net(["extractor", "classifier", "generator", "discriminator"], True)
        self.EC_optimizer = optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.lr,
                                       weight_decay=args.weight_decay)
        self.D_optimizer = optim.Adam(self.get_params(["discriminator"]), lr=args.lr, betas=(0.5, 0.999))
        self.G_optimizer = optim.Adam(self.get_params(["generator"]), lr=args.lr, betas=(0.5, 0.999))

        self.net.to(device)

        self.BCE_criterion = nn.BCELoss().to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.COS_criterion = nn.CosineSimilarity().to(device)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model[0].train()
        self.model[0].train()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        start_time = time.time()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)
        

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
        

        logger.info("Training Client %2d's EC Network Start!" % self.id)
        EC_loss_meter = AvgMeter()
        EG_distance_meter = AvgMeter()

        for epoch in range(args.local_epoch):
            EC_loss_meter.reset()
            EG_distance_meter.reset()

            self.frozen_net(["extractor", "classifier"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)

                self.EC_optimizer.zero_grad()

                E = self.net["extractor"](x)
                EC = self.net["classifier"](E)
                EC_loss = self.CE_criterion(EC, y)

                G = self.net["generator"](z, y).detach()
                if args.distance == "mse":
                    EG_distance = self.MSE_criterion(E, G)
                elif args.distance == "cos":
                    EG_distance = 1 - self.COS_criterion(E, G).mean()
                elif args.distance == "none":
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
            EC_acc = self.local_val()
            logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EG_distance:%2.6f, EC_acc:%2.6f" % (
                self.id, epoch, EC_loss, EG_distance, EC_acc))

        logger.info("Training Client %2d's DG Network Start!" % self.id)
        ED_loss_meter = AvgMeter()
        GD_loss_meter = AvgMeter()
        G_loss_meter = AvgMeter()

        for epoch in range(args.gan_epoch):
            ED_loss_meter.reset()
            GD_loss_meter.reset()
            G_loss_meter.reset()

            self.frozen_net(["generator", "discriminator"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)
                ones = torch.ones((x.size(0), 1)).to(device)
                zeros = torch.zeros((x.size(0), 1)).to(device)

                # train discriminator
                def train_discriminator(x, y, z):
                    self.D_optimizer.zero_grad()

                    E = self.net["extractor"](x)
                    ED = self.net["discriminator"](E.detach(), y)
                    ED_loss = self.BCE_criterion(ED, ones)

                    G = self.net["generator"](z, y)
                    GD = self.net["discriminator"](G.detach(), y)
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
                    G = self.net["generator"](z, y)
                    GD = self.net["discriminator"](G, y)
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
            logger.info("Client:[%2d], Epoch:[%2d], ED_loss:%2.6f, GD_loss:%2.6f, G_loss:%2.6f, GC_acc:%2.6f"
                        % (self.id, epoch, ED_loss, GD_loss, G_loss, GC_acc))


    def compute_gan_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch in range(100):
                y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
                z = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)
                feat = self.net["generator"](z, y)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += args.batch_size
        return (correct / total).item()
    
    def local_test(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.testloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()

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

    def local_train(self, current_round):
        print("Training Client %2d's EC Network Start!" % self.id)
        EC_loss_meter = AvgMeter()
        EG_distance_meter = AvgMeter()

        for epoch in range(args.local_epoch):
            EC_loss_meter.reset()
            EG_distance_meter.reset()

            self.frozen_net(["extractor", "classifier"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)

                self.EC_optimizer.zero_grad()

                E = self.net["extractor"](x)
                EC = self.net["classifier"](E)
                EC_loss = self.CE_criterion(EC, y)

                G = self.net["generator"](z, y).detach()
                if args.distance == "mse":
                    EG_distance = self.MSE_criterion(E, G)
                elif args.distance == "cos":
                    EG_distance = 1 - self.COS_criterion(E, G).mean()
                elif args.distance == "none":
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
            EC_acc = self.local_val()
            logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EG_distance:%2.6f, EC_acc:%2.6f" % (
                self.id, epoch, EC_loss, EG_distance, EC_acc))

        logger.info("Training Client %2d's DG Network Start!" % self.id)
        ED_loss_meter = AvgMeter()
        GD_loss_meter = AvgMeter()
        G_loss_meter = AvgMeter()

        for epoch in range(args.gan_epoch):
            ED_loss_meter.reset()
            GD_loss_meter.reset()
            G_loss_meter.reset()

            self.frozen_net(["generator", "discriminator"], False)

            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                z = torch.randn(x.size(0), args.noise_dim, 1, 1).to(device)
                ones = torch.ones((x.size(0), 1)).to(device)
                zeros = torch.zeros((x.size(0), 1)).to(device)

                # train discriminator
                def train_discriminator(x, y, z):
                    self.D_optimizer.zero_grad()

                    E = self.net["extractor"](x)
                    ED = self.net["discriminator"](E.detach(), y)
                    ED_loss = self.BCE_criterion(ED, ones)

                    G = self.net["generator"](z, y)
                    GD = self.net["discriminator"](G.detach(), y)
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
                    G = self.net["generator"](z, y)
                    GD = self.net["discriminator"](G, y)
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
            logger.info("Client:[%2d], Epoch:[%2d], ED_loss:%2.6f, GD_loss:%2.6f, G_loss:%2.6f, GC_acc:%2.6f"
                        % (self.id, epoch, ED_loss, GD_loss, G_loss, GC_acc))