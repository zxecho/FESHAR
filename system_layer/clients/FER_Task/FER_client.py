import copy
import torch
import torch.nn as nn
import numpy as np
import time
from system_layer.clients.client_base import Client

from sklearn.preprocessing import label_binarize
from sklearn import metrics


class FERclient(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # differential privacy
        # if self.privacy:
        #     check_dp(self.model)
        #     initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma,
        # )

    def train(self, **kwargs):
        if "global_epoch" in kwargs.keys():
            global_epoch = kwargs['global_epoch']
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

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
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                if self.privacy:
                    # dp_step(self.optimizer, i, len(trainloader))
                    pass
                else:
                    self.optimizer.step()

        # if global_epoch % self.lr_decay_every == 0:
        #     self.learning_rate_scheduler.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # if self.privacy:
        #     res, DELTA = get_dp_params(self.optimizer)
        #     print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
