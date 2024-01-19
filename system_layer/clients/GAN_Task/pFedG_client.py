import copy

import torch
import numpy as np
import time
from tqdm import tqdm
from system_layer.clients.GAN_Task.GAN_client_base import GAN_client
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params
from system_layer.training_utils import AvgMeter, add_gaussian_noise


# 包含三个模块，特征提取器网络E（不共享），分类器网络C（不共享），判别器网络D（共享），生成器网络G（共享）
class pFedG_Clinet(GAN_client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super(pFedG_Clinet, self).__init__(args, id, train_samples, test_samples, **kwargs)

        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.model["extractor"] = copy.deepcopy(args.model['extractor'])   # E
        self.model["classifier"] = copy.deepcopy(args.model['classifier'])   # C
        # self.model["generator"] = copy.deepcopy(args.model['generator'])   # E
        # self.model["discriminator"] = copy.deepcopy(args.model['discriminator'])   # C
        self.model.to(self.device)

        self.trainloader = self.load_train_data()
        for x, y in self.trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model['extractor'](x).detach()
            break
        self.feature_dim = rep.shape[1]

        # self.frozen_net(modules=["extractor", "classifier", "generator", "discriminator"], frozen=True)

        self.optimizer = torch.optim.AdamW(self.get_params(["extractor"]), lr=args.local_learning_rate)

        self.sample_per_class = torch.zeros(self.num_classes)
        for x, y in self.trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        self.qualified_labels = []

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
        start_time = time.time()

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, self.trainloader, self.dp_sigma)

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # log EC & EG loss
        EC_loss_meter = AvgMeter()
        EG_distance_meter = AvgMeter()

        EC_loss = EC_loss_meter.get()
        EG_distance = EG_distance_meter.get()
        # EC_acc = self.local_test()
        print("\n >>>>>>>>>>>> Client:[%2d] Start local training!<<<<<<<<<<<<<<<" % self.id)

        data_length = len(self.trainloader)

        # training local classifier with Generator
        self.train_local_C_with_G(current_round)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # if self.privacy:
        #     res, DELTA = get_dp_params(self.optimizer)
        #     print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def train_local_C_with_G(self, current_round):
        # 冻结生成器，判别器参数
        self.frozen_net(["generator", "discriminator"], True)
        start_time = time.time()

        max_local_epochs = self.local_steps
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # 生成器的损失函数权重
        p = min(current_round / 20, 1.)
        gamma = 2 / (1 + np.exp(-10 * p)) - 1

        for step in tqdm(range(max_local_epochs)):
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                feat = self.model['extractor'](x)
                output = self.model['classifier'](feat)
                loss = self.loss(output, y)

                labels = np.random.choice(self.qualified_labels, self.batch_size)
                labels = torch.LongTensor(labels).to(self.device)
                z = self.model['generator'](labels)
                loss += gamma * self.loss(self.model['classifier'](z), labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        self.model.eval()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in self.trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                feat = self.model["extractor"](x)
                output = self.model["classifier"](feat)
                loss = self.loss(output, y)

                labels = np.random.choice(self.qualified_labels, self.batch_size)
                labels = torch.LongTensor(labels).to(self.device)
                z = self.model["generator"](labels)
                loss += self.loss(self.model['classifier'](z), labels)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def test_metrics(self):
    #     testloader = self.load_test_data()
    #     correct, total = 0, 0
    #
    #     self.model.eval()
    #
    #     with torch.no_grad():
    #         for batch, (x, y) in enumerate(testloader):
    #             if self.args.add_noise:
    #                 x += add_gaussian_noise(x, mean=0., std=self.noise_std)
    #             x = x.to(self.device)
    #             y = y.to(self.device)
    #             feat = self.model["extractor"](x)
    #             pred = self.model["classifier"](feat)
    #             correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
    #             total += x.size(0)
    #     return correct.item(), total, 0
