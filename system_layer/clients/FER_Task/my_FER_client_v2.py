import copy
import torch
import torch.nn as nn
import numpy as np
import time
from system_layer.clients.client_base import Client
from system_layer.training_utils import clip_gradient

from sklearn.preprocessing import label_binarize
from sklearn import metrics


class FER_Client_v2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        # differential privacy
        # if self.privacy:
        #     check_dp(self.model)
        #     initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma,
        # )
        self.layer_idx = args.layer_idx

    def train(self):
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
                self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, global_model):
        if isinstance(global_model, list):
            params_g = global_model
        else:
            params_g = list(global_model.parameters())
        params = list(self.model.parameters())

        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

    def get_uploaded_params(self):
        params = list(self.model.parameters())
        return params[:-self.layer_idx]
