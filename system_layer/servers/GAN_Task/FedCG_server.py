import time
import torch
from torch import nn
from system_layer.clients.GAN_Task.FedCG_client import FedCG_client
from system_layer.servers.GAN_Task.GAN_server_base import GAN_server
from training_utils import frozen_net


class FedCG(GAN_server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(FedCG_client)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # server aggregation with KD
        # create the global federated model
        self.global_net = nn.ModuleDict()
        self.global_net["generator"] = Generator()
        self.global_net["classifier"] = Classifier()
        self.frozen_net(["generator", "classifier"], True)
        self.global_net.to(device)
        # define the optimizers
        self.GC_optimizer = optim.Adam(self.get_params(["generator", "classifier"]), lr=args.lr)
        self.KL_criterion = nn.KLDivLoss(reduction="batchmean").to(args.device)
        self.CE_criterion = nn.CrossEntropyLoss().to(args.device)

    def train(self):
        self.frozen_net(["generator", "classifier"], False)

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # -----------------------------------
            # train the server aggregation with KD 
            for batch in range(args.global_iter_per_epoch):
                y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
                z = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)

                self.GC_optimizer.zero_grad()

                global_feat = self.global_net["generator"](z, y)
                global_pred = self.global_net["classifier"](global_feat)
                q = torch.log_softmax(global_pred, -1)
                p = 0
                # each selected client starts training
                for i, client in enumerate(self.clients):
                    local_feat = client.net["generator"](z, y)
                    local_pred = client.net["classifier"](local_feat)
                    p += self.weights[i] * local_pred
                p = torch.softmax(p, -1).detach()
                distill_loss = self.KL_criterion(q, p)

                distill_loss.backward()
                self.GC_optimizer.step()
                distill_loss_meter.update(distill_loss.item())

            distill_loss = distill_loss_meter.get()
            logger.info("Server Epoch:[%2d], distill_loss:%2.6f" % (epoch, distill_loss))

        self.frozen_net(["generator", "classifier"], True)

        after_distill_acc = self.compute_aggregate_acc()
            # -----------------------------------

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            # if self.dlg_eval and i % self.dlg_gap == 0:
            #     self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def compute_aggregate_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for client in self.clients:
                for x, y in client.valloader:
                    if args.add_noise:
                        x = add_gaussian_noise(x, mean=0., std=client.noise_std)
                    x = x.to(device)
                    y = y.to(device)
                    feat = client.net["extractor"](x)
                    pred = self.global_net["classifier"](feat)
                    correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                    total += x.size(0)
        return (correct / total).item()