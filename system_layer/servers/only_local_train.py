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

                if step % self.args.eval_gap == 0:
                    train_loss, _ = self.train_metrics()
                    test_acc, _, test_auc = self.test_metrics()

                    train_losses_list.append(train_loss)
                    test_correct_list.append(test_acc)
                    test_auc_list.append(test_auc)

                pbar.update(1)
                pbar.set_postfix(train_loss=train_loss, test_acc=test_acc, test_auc=test_auc)

        pbar.close()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return train_losses_list, test_correct_list, test_auc_list

    # evaluate selected clients
    def evaluate(self, acc=None, auc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

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
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def save_results(self):
        dataset_dir = self.dataset.split('/')
        if len(dataset_dir) > 1:
            dataset = dataset_dir[-1]
        algo = dataset + "_" + self.algorithm
        result_path = "./results/{}/".format(self.save_folder_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

            plot_resutls((self.rs_test_acc, self.rs_test_auc, self.rs_train_loss), result_path,
                         algo)