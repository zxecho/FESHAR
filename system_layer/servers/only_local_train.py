import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

from system_layer.clients.client_base import Client
from system_layer.servers.server_base import Server
from infrastructure_layer.plot_data import plot_resutls


class OnlyLocalTrain_server(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, OnlyLocalClient)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # log global test results
        self.each_client_max_test_acc = []
        self.each_client_max_test_auc = []

    def select_clients(self):
        # 选择所有的客户端
        selected_clients = list(self.clients)

        return selected_clients

    def train(self):
        self.selected_clients = self.select_clients()
        self.send_models()

        for client in self.selected_clients:
            s_t = time.time()
            self.rs_train_loss, self.rs_test_acc, self.rs_test_auc = client.train()
            self.each_client_max_test_acc.append(max(self.rs_test_acc))
            self.each_client_max_test_auc.append(max(self.rs_test_auc))

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'Clinet {} time cost:{}'.format(client.id, self.Budget[-1]), '-' * 25)

            self.save_results(client_id=client.id)
        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))


class OnlyLocalClient(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.args = args
        self.client_id = id
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma,
        )

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()
        # log evaluate info
        train_losses_list = []
        test_correct_list = []
        test_auc_list = []

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.args.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)
        with tqdm(range(max_local_steps)) as pbar:
            pbar.set_description("Clinet {}".format(self.client_id))
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

                if self.learning_rate_decay and step % self.args.lr_decay_every == 0:
                    self.learning_rate_scheduler.step()
                # evaluating
                if step % self.args.eval_gap == 0:
                    train_loss, train_num_samples = self.train_metrics()
                    test_acc, test_num_samples, test_auc = self.test_metrics()

                    # log train metrics
                    train_loss = train_loss * 1.0 / train_num_samples
                    train_losses_list.append(train_loss)
                    # log test metrics
                    test_acc = test_acc * 1.0 / test_num_samples
                    # test_auc = test_auc * 1.0 / test_num_samples
                    test_correct_list.append(test_acc)
                    test_auc_list.append(test_auc)

                pbar.update(1)
                pbar.set_postfix(train_loss=train_loss, test_acc=test_acc, test_auc=test_auc)

        pbar.close()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return train_losses_list, test_correct_list, test_auc_list
