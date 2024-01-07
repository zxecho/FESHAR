import time
import numpy as np
import torch
import copy
from tqdm import tqdm

from system_layer.clients.GAN_Task.pFedG_client import pFedG_Clinet
from system_layer.servers.GAN_Task.GAN_server_base import GAN_server
from algo_layer.models.GAN_modules import Generator, Classifier, Extractor
from system_layer.training_utils import add_gaussian_noise, AvgMeter


class pFedG_server(GAN_server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        self.set_slow_clients()
        self.set_clients(pFedG_Clinet)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.global_model["extractor"] = Extractor(input_channels=args.image_channel)  # E
        # self.global_model["classifier"] = Classifier(n_classes=self.num_classes)  # C
        # self.global_model["generator"] = Generator(n_classes=args.num_classes, latent_dim=args.noise_dim,
        #                                            feature_num=self.clients[0].feature_dim)
        self.global_model = copy.deepcopy(args.model)
        self.global_model.to(self.device)
        # ============ optimizer of server generator ==================
        self.generative_optimizer = torch.optim.AdamW(
            params=self.global_model['generator'].parameters(),
            lr=args.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=args.learning_rate_decay_gamma)
        self.loss = torch.nn.CrossEntropyLoss()
        self.GC_criterion = torch.nn.CrossEntropyLoss()

        # define the optimizers
        self.GC_optimizer = torch.optim.Adam(self.get_params(["generator", "classifier"]), lr=args.global_learning_rate)
        self.KL_criterion = torch.nn.KLDivLoss(reduction="batchmean").to(args.device)

        # ============= statistic the labels ===============
        self.qualified_labels = []
        for client in self.clients:
            for yy in range(self.num_classes):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
        for client in self.clients:
            client.qualified_labels = self.qualified_labels

        self.localize_feature_extractor = args.localize_feature_extractor
        # if self.localize_feature_extractor:
        #     self.global_model['classifier'] = copy.deepcopy(args.model['classifier'])
        # self.load_model()
        self.Budget = []

    def train(self):

        self.distill_loss_meter = AvgMeter()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models(['extractor', 'generator'])

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train(current_round=i)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # 只有在训练完所有客户端之后，才会计算本地训练准确率
            self.frozen_net(["generator", "classifier"], False)

            self.train_generator()
            # self.aggregate_parameters_with_KD(i)
            self.aggregate_parameters(['extractor'])

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # 只有在训练完所有客户端之后，才会计算本地训练准确率
        self.frozen_net(["generator", "classifier"], True)

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def train_generator(self):
        # self.global_model['generator'].train()
        # self.frozen_net(['generator', 'classifier'], False)

        for _ in tqdm(range(self.gan_server_epochs)):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            # z = torch.randn(self.batch_size, self.args.noise_dim, 1, 1).to(self.device)
            # gen_data = self.global_model["generator"](z, labels)
            gen_data = self.global_model["generator"](labels)

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                if self.localize_feature_extractor:
                    logits += model(gen_data) * w
                else:
                    logits += model['classifier'](gen_data) * w

            # Global classifier model loss
            global_pred = self.global_model['classifier'](gen_data)
            GC_loss = self.GC_criterion(global_pred, labels)

            self.generative_optimizer.zero_grad()
            loss = self.loss(logits, labels) + GC_loss
            loss.backward()
            self.generative_optimizer.step()

        self.generative_learning_rate_scheduler.step()
    def aggregate_parameters_with_KD(self, current_round):
        # 统计loss
        self.distill_loss_meter.reset()
        # train the server aggregation with KD
        for batch in range(self.args.global_iter_per_epoch):
            # y = torch.randint(0, self.args.num_classes, (self.args.batch_size,)).to(self.device)
            # z = torch.randn(self.args.batch_size, self.args.noise_dim, 1, 1).to(self.device)
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)

            self.GC_optimizer.zero_grad()

            global_feat = self.global_model["generator"](labels)
            global_pred = self.global_model["classifier"](global_feat)
            q = torch.log_softmax(global_pred, -1)
            p = 0
            # each selected client starts training
            for i, client in enumerate(self.selected_clients):
                local_feat = client.model["generator"](labels)
                local_pred = client.model["classifier"](local_feat)
                p += self.uploaded_weights[i] * local_pred
            p = torch.softmax(p, -1).detach()
            distill_loss = self.KL_criterion(q, p)

            distill_loss.backward()
            self.GC_optimizer.step()
            self.distill_loss_meter.update(distill_loss.item())

        distill_loss = self.distill_loss_meter.get()
        self.logger.info("Server Epoch:[%2d], distill_loss:%2.6f" % (current_round, distill_loss))