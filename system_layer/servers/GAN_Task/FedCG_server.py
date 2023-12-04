import time
import torch
from torch import nn
from system_layer.clients.GAN_Task.FedCG_client import FedCG_client
from system_layer.servers.GAN_Task.GAN_server_base import GAN_server
from algo_layer.models.GAN_modules import Classifier
from system_layer.training_utils import add_gaussian_noise, AvgMeter


class FedCG(GAN_server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(FedCG_client)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        # server aggregation with KD
        # create the global federated model
        self.global_model["classifier"] = Classifier(n_classes=self.num_classes)
        self.frozen_net(["generator", "classifier"], True)
        self.global_model.to(args.device)
        # define the optimizers
        self.GC_optimizer = torch.optim.Adam(self.get_params(["generator", "classifier"]), lr=args.global_learning_rate)
        self.KL_criterion = nn.KLDivLoss(reduction="batchmean").to(args.device)
        self.CE_criterion = nn.CrossEntropyLoss().to(args.device)

        self.distill_loss_meter = None
        self.distill_acc_meter = None

        # record best val acc
        self.client_best_test_accs = [0.] * self.args.num_clients
        self.client_test_accs = [0.] * self.args.num_clients
        self.client_best_rounds = [-1] * self.args.num_clients

    def compute_aggregate_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for client in self.clients:
                for x, y in client.load_train_data():
                    if self.args.add_noise:
                        x = add_gaussian_noise(x, mean=0., std=client.noise_std)
                    x = x.to(self.args.device)
                    y = y.to(self.args.device)
                    feat = client.model["extractor"](x)
                    pred = self.global_model["classifier"](feat)
                    correct += torch.sum(torch.tensor(torch.argmax(pred, dim=1) == y, dtype=torch.float))
                    total += x.size(0)
        return (correct / total).item()

    def get_params(self, models):
        params = []
        for model in models:
            params.append({"params": self.global_model[model].parameters()})

        return params

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.global_model[model].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.global_model[model].eval()
            else:
                self.global_model[model].train()

    def train(self):
        self.frozen_net(["generator", "classifier"], False)

        self.logger.info("Training Server's Network Start!")
        self.distill_loss_meter = AvgMeter()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # for test local model
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate(current_round=i, test_accs=self.client_test_accs,
                              best_test_accs=self.client_best_test_accs,
                              best_rounds=self.client_best_rounds)

            # each client local train
            for client in self.selected_clients:
                client.train(current_round=i)

            self.receive_models()
            self.aggregate_parameters()
            # 服务器端聚合算法
            self.aggregate_parameters_with_KD(current_round=i)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        self.frozen_net(["generator", "classifier"], True)

        after_distill_acc = self.compute_aggregate_acc()
        self.logger.info("after distill client acc:%2.6f" % after_distill_acc)

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # self.save_results()
        self.save_global_model()

    def aggregate_parameters_with_KD(self, current_round):
        # 统计loss
        self.distill_loss_meter.reset()
        # train the server aggregation with KD
        for batch in range(self.args.global_iter_per_epoch):
            y = torch.randint(0, self.args.num_classes, (self.args.batch_size,)).to(self.device)
            z = torch.randn(self.args.batch_size, self.args.noise_dim, 1, 1).to(self.device)

            self.GC_optimizer.zero_grad()

            global_feat = self.global_model["generator"](z, y)
            global_pred = self.global_model["classifier"](global_feat)
            q = torch.log_softmax(global_pred, -1)
            p = 0
            # each selected client starts training
            for i, client in enumerate(self.selected_clients):
                local_feat = client.model["generator"](z, y)
                local_pred = client.model["classifier"](local_feat)
                p += self.uploaded_weights[i] * local_pred
            p = torch.softmax(p, -1).detach()
            distill_loss = self.KL_criterion(q, p)

            distill_loss.backward()
            self.GC_optimizer.step()
            self.distill_loss_meter.update(distill_loss.item())

        distill_loss = self.distill_loss_meter.get()
        self.logger.info("Server Epoch:[%2d], distill_loss:%2.6f" % (current_round, distill_loss))

    # TODO: 修改测试评估函数
    def evaluate(self, current_round, test_accs=None, best_test_accs=None, best_rounds=None):
        # client test
        for i, client in enumerate(self.clients):
            val_acc = client.local_test()
            if val_acc > best_test_accs[i]:
                best_test_accs[i] = val_acc
                test_accs[i] = client.local_test()
                best_rounds[i] = current_round

        # round result
        self.logger.info("communication round:%2d, after local train result:" % current_round)
        for i in range(self.args.num_clients):
            self.logger.info("client:%2d, test acc:%2.6f, best epoch:%2d" % (i, test_accs[i], best_rounds[i]))
