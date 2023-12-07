import torch
import numpy as np
import time
from tqdm import tqdm
from system_layer.clients.GAN_Task.GAN_client_base import GAN_client
from algo_layer.models.GAN_modules import Extractor, Classifier
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params
from system_layer.training_utils import AvgMeter, add_gaussian_noise


# 包含三个模块，特征提取器网络E（不共享），分类器网络C（不共享），判别器网络D（共享），生成器网络G（共享）
class ZLGAN_Clinet(GAN_client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.model["extractor"] = Extractor(input_channels=args.image_channel)  # E
        self.model["classifier"] = Classifier(n_classes=self.num_classes)  # C

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

        self.EC_optimizer = torch.optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.local_learning_rate,
                                       weight_decay=args.weight_decay)
        self.BCE_criterion = torch.nn.BCELoss().to(self.device)
        self.CE_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.MSE_criterion = torch.nn.MSELoss().to(self.device)
        self.COS_criterion = torch.nn.CosineSimilarity().to(self.device)

        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

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
        print("\n Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EG_distance:%2.6f" % (
            self.id, current_round, EC_loss, EG_distance))

        data_length = len(self.trainloader)

        # 使用进度条代替循环, 训练本地的GAN模型
        for step in range(self.args.gan_client_epoch):
            # 将特征提取器和分类器解冻
            self.frozen_net(["discriminator", "generator"], False)
            # self.model.train()

            with tqdm(range(data_length)) as tbar:
                for i, (x, y) in enumerate(self.trainloader):

                    if self.args.add_noise:
                        x += add_gaussian_noise(x, mean=0., std=self.noise_std)

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    batch_size = x.shape[0]

                    # sample noise and labels as generator input
                    # z = torch.randn(batch_size, self.latent_dim, device=self.device)      # old version
                    z = torch.randn(x.size(0), self.args.noise_dim, 1, 1).to(self.device)
                    gen_labels = torch.randint(0, self.args.num_classes, (x.size(0),)).to(self.device)
                    # Adversarial ground truths
                    valid = torch.ones((x.size(0), 1), requires_grad=False).to(self.device)
                    fake = torch.zeros((x.size(0), 1), requires_grad=False).to(self.device)

                    # ========== train discriminator ==========
                    self.d_optimizer.zero_grad()

                    feature_E = self.model['extractor'](x)
                    ED = self.model["discriminator"](feature_E.detach(), y)
                    ED_loss = self.BCE_criterion(ED, valid)

                    gen_imgs = self.model["generator"](z, y)
                    gen_imgs_d = self.model["discriminator"](gen_imgs.detach(), y)
                    GD_loss = self.BCE_criterion(gen_imgs_d, fake)

                    D_loss = ED_loss + GD_loss

                    # Calculate discriminator accuracy
                    real_aux = self.model['classifier'](feature_E)
                    fake_aux = self.model['classifier'](gen_imgs.detach())
                    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([y.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                    D_loss.backward()
                    self.d_optimizer.step()

                    # ========== train generator ==========
                    self.g_optimizer.zero_grad()

                    # gen loss
                    gen_imgs = self.model["generator"](z, y)
                    GD = self.model["discriminator"](gen_imgs, y)
                    G_loss = self.BCE_criterion(GD, valid)

                    G_loss.backward()
                    self.g_optimizer.step()

                    # ============= print training information =============

                    # 使用tqdm打印一下信息
                    tbar.set_description(
                        "[Local GAN Epoch {}/{}] [Batch {}/{}] [D loss: {}, acc: {}] [G loss: {}]"
                        .format(step, self.args.gan_client_epoch, i,
                                data_length, D_loss.item(), 100 * d_acc, G_loss.item())
                    )
                    tbar.update(1)

        self.frozen_net(["generator", "discriminator"], True)

        # training local classifier with GAN
        self.train_local_C_with_G(current_round, max_local_steps)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # if self.privacy:
        #     res, DELTA = get_dp_params(self.optimizer)
        #     print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

    def train_local_C_with_G(self, current_round, max_local_steps):

        data_length = len(self.trainloader)
        self.frozen_net(["extractor", "classifier"], False)
        # local classifier training progress
        for step in range(max_local_steps):
            with tqdm(range(data_length)) as tbar:
                for i, (x, y) in enumerate(self.trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    z = torch.randn(x.size(0), self.args.noise_dim, 1, 1).to(self.device)

                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    # compute the lcoal training loss
                    self.EC_optimizer.zero_grad()

                    E = self.model["extractor"](x)
                    EC = self.model["classifier"](E)
                    EC_loss = self.CE_criterion(EC, y)

                    G = self.model["generator"](z, y).detach()

                    if self.args.distance == "mse":
                        EG_distance = self.MSE_criterion(E, G)
                    elif self.args.distance == "cos":
                        EG_distance = 1 - self.COS_criterion(E, G).mean()
                    elif self.args.distance == "none":
                        EG_distance = 0
                    p = min(current_round / 50, 1.)
                    gamma = 2 / (1 + np.exp(-10 * p)) - 1
                    EG_distance = gamma * EG_distance

                    (EC_loss + EG_distance).backward()
                    self.EC_optimizer.step()

                    # output = self.model(x)
                    # loss = self.loss(output, y)
                    #
                    # labels = np.random.choice(self.qualified_labels, self.batch_size)
                    # labels = torch.LongTensor(labels).to(self.device)
                    # z = self.model['generator'](labels)
                    # loss += self.loss(self.model['classifier'](z), labels)
                    #
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()

                    # 使用tqdm打印一下信息
                    tbar.set_description(
                        "[Local C Epoch {}/{}] [Batch {}/{}] [EC_loss: {}, EG_distance: {}]"
                        .format(step, max_local_steps, i, data_length, EC_loss.item(), EG_distance.item())
                    )
                    tbar.update(1)

        self.frozen_net(["extractor", "classifier"], True)

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
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def test_metrics(self):
        testloader = self.load_test_data()
        correct, total = 0, 0

        self.model.eval()

        with torch.no_grad():
            for batch, (x, y) in enumerate(testloader):
                if self.args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(self.device)
                y = y.to(self.device)
                feat = self.model["extractor"](x)
                pred = self.model["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return correct.item(), total, 0
