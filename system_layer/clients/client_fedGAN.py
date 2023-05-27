import torch
import torch.nn as nn
import numpy as np
import time
from system_layer.clients.client_base import Client


# from algo_layer.privacy import *


class clientGAN(Client):

    def __init__(self):
        pass

    def train_metrics(self):
        pass

    def test_metrics(self):
        pass

    def evaluate(self):
        pass


def Local_ACGAN_Train(args, localGModel, localDModel, globalGModel, globalDModel, dataset_loader, dataset_lb_stat,
                      local_g_lr=1e-4, local_d_lr=1e-4,
                      save_path='', name='', main_epoch=0):
    """
    Use Aux Classifier GAN
    :param local_d_lr:
    :param local_g_lr:
    :param args:
    :param globalGModel:
    :param globalDModel:
    :param dataset_loader:
    :param dataset_lb_stat:
    :param save_path:
    :param name:
    :param fixed_z:
    :param main_epoch:
    :return:
    """

    # 保存路径
    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    # 有多少类数据
    n_classes = len(args.dataset_label[args.dataset])

    # 用于保存模型
    file_name = save_path.split('/')
    file_name = file_name[2] + '/' + file_name[3]

    # --------------- 先训练conditional GAN网络 ---------------
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # adversarial_loss = torch.nn.MSELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    # F.nll_loss

    # labels_processing
    # real_labels = 0.7 + 0.5 * torch.rand(n_classes, device=args.device)
    # fake_labels = 0.3 * torch.rand(n_classes, device=args.device)

    # Optimizers
    optimizer_G = torch.optim.Adam(localGModel.parameters(), lr=local_g_lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(localDModel.parameters(), lr=local_d_lr, betas=(args.b1, args.b2))

    # optimizer_G = torch.optim.SGD(globalGModel.parameters(), lr=local_g_lr, momentum=0.95)
    # optimizer_D = torch.optim.SGD(globalDModel.parameters(), lr=local_d_lr)

    # ----------
    #  Training
    # ----------

    # 用于记录训练过程
    local_g_training_losses_list = []
    local_d_training_losses_list = []

    globalG_weight_collector = list(globalGModel.parameters())
    globalD_weight_collector = list(globalDModel.parameters())

    with tqdm(range(args.local_ep)) as tq:
        # with tqdm(enumerate(dataset_loader), total=args.local_ep) as tq:
        # 记录评估
        emd_list = []
        c_scores_list = []
        best_c_acc = 0
        tq.set_description('{} training'.format(name))
        localGModel.train()
        localDModel.train()

        # for i, (inputs, targets) in tq:
        #
        #     if i == args.local_ep:
        #         break

        # 用于记录训练过程中的loss变化
        local_g_training_loss = 0
        local_d_training_loss = 0

        for epoch in tq:

            tau = 0

            inputs, targets = next(iter(dataset_loader))
            batch_size = inputs.shape[0]
            tau += 1

            # Configure input
            # real_imgs = Variable(imgs.type(Tensor))
            real_imgs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Not smooth processing
            real_label = 1.0
            fake_label = 0.0

            fake_class_labels = (n_classes - 1) * torch.ones((batch_size,), dtype=torch.long, device=args.device)
            validity_label = torch.full((batch_size, 1), real_label, device=args.device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            noise = torch.randn(batch_size, args.latent_dim, device=args.device)
            # sample_labels = torch.randint(0, n_classes, (batch_size,), device=args.device, dtype=torch.long)
            random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
            sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)
            fake_labels = sample_labels.view(-1, 1)

            # Generate a batch of images
            gen_imgs = localGModel(noise, sample_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = localDModel(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, validity_label) + auxiliary_loss(pred_label, sample_labels))

            # for fedprox
            if args.FedPorx:
                fed_prox_reg = 0.0
                for param_index, param in enumerate(localGModel.parameters()):
                    fed_prox_reg += ((args.fedprox_mu / 2) * torch.norm(
                        (param - globalG_weight_collector[param_index])) ** 2)
                g_loss += fed_prox_reg

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred, real_aux = localDModel(real_imgs)
            d_real_loss = (adversarial_loss(real_pred, validity_label) + auxiliary_loss(real_aux, targets)) / 2

            # Loss for fake images
            fake = torch.full((batch_size, 1), fake_label, device=args.device)
            fake_pred, fake_aux = localDModel(gen_imgs.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, sample_labels)) / 2
            # d_fake_loss = adversarial_loss(fake_pred, fake) / 2     # Mine algo

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            # gt = np.concatenate([targets.data.cpu().numpy(), sample_labels.data.cpu().numpy()], axis=0)
            # d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            if args.FedPorx & args.if_Fed_train_D:
                fed_prox_reg = 0.0
                for param_index, param in enumerate(localDModel.parameters()):
                    fed_prox_reg += ((args.fedprox_mu / 2) * torch.norm(
                        (param - globalD_weight_collector[param_index])) ** 2)
                g_loss += fed_prox_reg

            d_loss.backward()
            optimizer_D.step()

            '''
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # For real data

            real_pvalidity, real_plabels = localDModel(real_imgs)

            errD_real_val = adversarial_loss(real_pvalidity, validity_label)
            errD_real_label = F.nll_loss(real_plabels, targets)
            # errD_real_label = auxiliary_loss(real_plabels, targets)
            errD_real = (errD_real_val + errD_real_label) * 0.5
            errD_real.backward()
            # D_x = real_pvalidity.mean().item()

            # For fake data
            random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
            sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)
            noise = torch.randn(batch_size, args.latent_dim, device=args.device)

            # Generate a batch of images
            gen_imgs = localGModel(noise, sample_labels)

            validity_label.fill_(fake_label)

            fake_pvalidity, fake_plabels = localDModel(gen_imgs.detach())

            errD_fake_val = adversarial_loss(fake_pvalidity, validity_label)
            errD_fake_label = F.nll_loss(fake_plabels, sample_labels)
            # errD_fake_label = auxiliary_loss(fake_plabels, sample_labels)
            errD_fake = 0.5 * (errD_fake_val) # + errD_fake_label)
            errD_fake.backward()
            # D_G_z1 = fake_pvalidity.mean().item()

            d_loss = errD_real + errD_fake

            # d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            noise = torch.randn(batch_size, args.latent_dim, device=args.device)
            # sample_labels = torch.randint(0, n_classes, (batch_size,), device=args.device, dtype=torch.long)
            random_sample_lbs = np.random.choice(dataset_lb_stat, size=batch_size)
            sample_labels = torch.as_tensor(random_sample_lbs, dtype=torch.int64, device=args.device)

            validity_label.fill_(real_label)

            # Generate a batch of images
            gen_imgs = localGModel(noise, sample_labels)

            fake_pvalidity, fake_plabels = localDModel(gen_imgs)

            errG_val = adversarial_loss(fake_pvalidity, validity_label)
            errG_label = F.nll_loss(fake_plabels, sample_labels)  # 错误 fake_class_labels
            # errG_label = auxiliary_loss(fake_plabels, sample_labels)

            # D_G_z2 = pvalidity.mean().item()

            g_loss = errG_val + errG_label

            # for fedprox
            if args.FedPorx:
                fed_prox_reg = 0.0
                for param_index, param in enumerate(localGModel.parameters()):
                    fed_prox_reg += ((args.fedprox_mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                g_loss += fed_prox_reg

            g_loss.backward()
            optimizer_G.step()
            '''

            local_d_training_loss += d_loss
            local_g_training_loss += g_loss

            # local_g_training_losses = local_g_training_loss.cpu().detach().numpy() / (i + 1)
            # local_d_training_losses = local_d_training_loss.cpu().detach().numpy() / (i + 1)

        # 本地模型评估
        # LocalG_eval(globalGModel, save_path, name, args.latent_dim, n_row=0, main_epoch=main_epoch)
        AC_GAN_eval(args, localGModel, save_path + '/GNet_eval/', name, n_classes=n_classes)

    local_d_training_loss = local_d_training_loss.cpu().detach().numpy()
    local_g_training_loss = local_g_training_loss.cpu().detach().numpy()

    # 保存联邦模型 LocalGNet & LocalDNet
    save_model(localGModel, file_name, 'Local_{}_G'.format(name))
    save_model(localDModel, file_name, 'Local_{}_D'.format(name))

    # For FedNova
    a_i = None
    if args.FedNova:
        a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
        global_model_para = globalGModel.state_dict()
        Gnet_para = localGModel.state_dict()
        Gnorm_grad = copy.deepcopy(globalGModel.state_dict())
        for key in Gnorm_grad:
            Gnorm_grad[key] = torch.true_divide(global_model_para[key] - Gnet_para[key], a_i)
        localGModel.load_state_dict(Gnorm_grad)

        if args.if_Fed_train_D:
            dlobal_model_para = globalDModel.state_dict()
            Dnet_para = localDModel.state_dict()
            Dnorm_grad = copy.deepcopy(globalDModel.state_dict())
            for key in Dnorm_grad:
                Dnorm_grad[key] = torch.true_divide(dlobal_model_para[key] - Dnet_para[key], a_i)
            localDModel.load_state_dict(Dnorm_grad)

    return localGModel.state_dict(), localDModel.state_dict(), \
        local_d_training_loss, local_g_training_loss, \
        (dataset_loader.dataset.number, a_i)
