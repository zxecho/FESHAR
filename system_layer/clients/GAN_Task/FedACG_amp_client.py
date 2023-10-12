import torch
import numpy as np
import time
from tqdm import tqdm
from system_layer.clients.GAN_Task.GAN_client_base import GAN_client
from algo_layer.models.GAN_modules import Extractor, Classifier
from algo_layer.privacy_protect_utils import initialize_dp, get_dp_params
from system_layer.training_utils import AvgMeter, add_gaussian_noise


# 包含三个模块，特征提取器网络E（不共享），分类器网络C（不共享），判别器网络D（共享），生成器网络G（共享）
class ACGAN_Clinet(GAN_client):

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.model["extractor"] = Extractor(input_channels=args.image_channel)  # E
        self.model["classifier"] = Classifier(n_classes=self.num_classes)  # C

        # self.frozen_net(modules=["extractor", "classifier", "generator", "discriminator"], frozen=True)

        self.model.to(self.device)

        self.EC_optimizer = torch.optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.local_learning_rate,
                                       weight_decay=args.weight_decay)
        self.BCE_criterion = torch.nn.BCELoss().to(self.device)
        self.CE_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.MSE_criterion = torch.nn.MSELoss().to(self.device)
        self.COS_criterion = torch.nn.CosineSimilarity().to(self.device)

        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

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
        trainloader = self.load_train_data()
        start_time = time.time()

        data_length = len(trainloader)

        # differential privacy
        if self.privacy:
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # log EC & EG loss
        EC_loss_meter = AvgMeter()
        EG_distance_meter = AvgMeter()
        # local classifier training progress
        for _ in tqdm(range(5)):
            self.frozen_net(["extractor", "classifier"], False)

            for i, (x, y) in enumerate(trainloader):
                if self.args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                z = torch.randn(x.size(0), self.args.noise_dim, 1, 1).to(self.device)
                # slow the training speed
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
                EC_loss_meter.update(EC_loss.item())
                EG_distance_meter.update(EG_distance.item())

            # self.frozen_net(["extractor", "classifier"], True)

        EC_loss = EC_loss_meter.get()
        EG_distance = EG_distance_meter.get()
        # EC_acc = self.local_test()
        print("\n Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EG_distance:%2.6f" % (
            self.id, current_round, EC_loss, EG_distance))

        # 使用进度条代替循环
        for step in range(max_local_steps):
            # 将特征提取器和分类器解冻
            self.frozen_net(["discriminator", "generator"], False)
            # self.model.train()

            with tqdm(range(data_length)) as tbar:
                for i, (x, y) in enumerate(trainloader):

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

                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.g_optimizer.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.model["generator"](z, gen_labels)

                    # Loss measures generator's ability to fool the discriminator
                    validity = self.model["discriminator"](gen_imgs, gen_labels)
                    pred_label = self.model['classifier'](gen_imgs)
                    g_loss = 0.5 * (self.adversarial_loss(validity, valid) + self.auxiliary_loss(pred_label, gen_labels))

                    g_loss.backward()
                    self.g_optimizer.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    self.d_optimizer.zero_grad()

                    feature_E = self.model['extractor'](x)
                    real_pred = self.model["discriminator"](feature_E, y)
                    real_aux = self.model['classifier'](feature_E)
                    # Loss for real images
                    d_real_loss = (self.adversarial_loss(real_pred, valid) + self.auxiliary_loss(real_aux, y)) / 2

                    # Loss for fake images
                    fake_pred = self.model["discriminator"](gen_imgs.detach(), gen_labels)
                    fake_aux = self.model['classifier'](gen_imgs.detach())
                    d_fake_loss = (self.adversarial_loss(fake_pred, fake) + self.auxiliary_loss(fake_aux, gen_labels)) / 2

                    # Total discriminator loss
                    d_loss = (d_real_loss + d_fake_loss) / 2

                    # Calculate discriminator accuracy
                    pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                    gt = np.concatenate([y.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                    d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                    d_loss.backward()
                    self.d_optimizer.step()

                    # ============= print training information =============

                    # 使用tqdm打印一下信息
                    tbar.set_description(
                        "[Epoch {}/{}] [Batch {}/{}] [D loss: {}, acc: {}] [G loss: {}]"
                        .format(step, self.local_steps, i, data_length, d_loss.item(), 100 * d_acc, g_loss.item())
                    )
                    tbar.update(1)

                    self.train_time_cost['num_rounds'] += 1
                    self.train_time_cost['total_cost'] += time.time() - start_time

                    # if self.privacy:
                    #     res, DELTA = get_dp_params(self.optimizer)
                    #     print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")
                    if self.privacy:
                        eps, DELTA = get_dp_params(privacy_engine)
                        print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            self.frozen_net(["generator", "discriminator"], True)

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
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
