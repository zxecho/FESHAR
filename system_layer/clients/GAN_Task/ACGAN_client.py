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
from system_layer.servers.server_base import Server
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params


class ACGAN_Clinet(GAN_client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

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

                batch_size = x.shape[0]

                # Adversarial ground truths
                valid = torch.tensor(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False, device=self.device)
                fake = torch.tensor(torch.FloatTensor(batch_size, 1).fill_(.0), requires_grad=False, device=self.device)

                # Configure input
                # real_imgs = torch.tensor(x, dtype=torch.float32)
                # labels = torch.tensor(y, dtype=torch.long)

                # -----------------
                #  Train Generator
                # -----------------

                self.g_optimizer.zero_grad()

                # Sample noise and labels as generator input
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                gen_labels = torch.tensor(np.random.randint(0, self.num_classes, batch_size), dtype=torch.long, device=self.device)

                # Generate a batch of images
                gen_imgs = self.g_model(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = self.d_model(gen_imgs)
                g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))

                g_loss.backward()
                self.g_optimizer.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.d_optimizer.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.d_model(x)
                d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, y)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.d_model(gen_imgs.detach())
                d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([y.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.d_optimizer.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (step, self.local_steps, i, len(trainloader), d_loss.item(), 100 * d_acc, g_loss.item())
                )

                self.train_time_cost['num_rounds'] += 1
                self.train_time_cost['total_cost'] += time.time() - start_time

                # if self.privacy:
                #     res, DELTA = get_dp_params(self.optimizer)
                #     print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
                if self.privacy:
                    eps, DELTA = get_dp_params(privacy_engine)
                    print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
