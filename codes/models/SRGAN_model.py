import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss


class SRGANModel(BaseModel):
    def name(self):
        return 'SRGANModel'

    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        train_opt = opt['train']

        self.input_L = self.Tensor()
        self.input_H = self.Tensor()
        self.input_ref = self.Tensor() # for Discriminator

        # define network and load pretrained models
        # Generator - SR network
        self.netG = networks.define_G(opt)
        self.load_path_G = opt['path']['pretrain_model_G']
        if self.is_train:
            self.need_pixel_loss = True
            self.need_feature_loss = True
            if train_opt['pixel_weight'] == 0:
                print('Set pixel loss to zero.')
                self.need_pixel_loss = False
            if train_opt['feature_weight'] == 0:
                print('Set feature loss to zero.')
                self.need_feature_loss = False
            assert self.need_pixel_loss or self.need_feature_loss, 'pixel and feature loss are both 0.'
            # Discriminator
            self.netD = networks.define_D(opt)
            self.load_path_D = opt['path']['pretrain_model_D']
            if self.need_feature_loss:
                self.netF = networks.define_F(opt, use_bn=False) # perceptual loss
        self.load() # load G and D if needed

        if self.is_train:
            # for wgan-gp
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = Variable(self.Tensor(1, 1, 1, 1))

            # define loss function
            # pixel loss
            pixel_loss_type = train_opt['pixel_criterion']
            if pixel_loss_type == 'l1':
                self.criterion_pixel = nn.L1Loss()
            elif pixel_loss_type == 'l2':
                self.criterion_pixel = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not recognized.' % pixel_loss_type)
            self.loss_pixel_weight = train_opt['pixel_weight']

            # feature loss
            feature_loss_type = train_opt['feature_criterion']
            if feature_loss_type == 'l1':
                self.criterion_feature = nn.L1Loss()
            elif feature_loss_type == 'l2':
                self.criterion_feature = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not recognized.' % feature_loss_type)
            self.loss_feature_weight = train_opt['feature_weight']

            # gan loss
            gan_type = train_opt['gan_type']
            self.criterion_gan = GANLoss(gan_type, real_label_val=1.0, fake_label_val=0.0, \
                    tensor=self.Tensor)
            self.loss_gan_weight = train_opt['gan_weight']

            # gradient penalty loss
            if train_opt['gan_type'] == 'wgan-gp':
                self.criterion_gp = GradientPenaltyLoss(tensor=self.Tensor)
            self.loss_gp_weight = train_opt['gp_weigth']

            if self.use_gpu:
                self.criterion_pixel.cuda()
                self.criterion_feature.cuda()
                self.criterion_gan.cuda()
                if train_opt['gan_type'] == 'wgan-gp':
                    self.criterion_gp.cuda()

            # initialize optimizers
            self.optimizers = [] # G and D
            # G
            self.lr_G = train_opt['lr_G']
            self.wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters(): # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARN: params [%s] will not optimize.' % k)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=self.lr_G, weight_decay=self.wd_G,\
                betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            self.lr_D = train_opt['lr_D']
            self.wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr_D, \
                weight_decay=self.wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            # initialize schedulers
            self.schedulers = []
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, volatile=False, need_HR=True):
        # LR
        input_L = data['LR']
        self.input_L.resize_(input_L.size()).copy_(input_L)
        self.real_L = Variable(self.input_L, volatile=volatile)

        if need_HR: # train or val
            input_H = data['HR']
            self.input_H.resize_(input_H.size()).copy_(input_H)
            self.real_H = Variable(self.input_H, volatile=volatile)  # in range [0,1]

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.input_ref.resize_(input_ref.size()).copy_(input_ref)
            self.real_ref = Variable(self.input_ref, volatile=volatile)  # in range [0,1]

    def optimize_parameters(self, step):
        # G
        self.optimizer_G.zero_grad()
        # forward G
        # self.real_L: leaf, not requires_grad; self.fake_H: no leaf, requires_grad
        self.fake_H = self.netG(self.real_L)

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.need_pixel_loss:
                loss_g_pixel = self.loss_pixel_weight * self.criterion_pixel(self.fake_H, self.real_H)
            # forward F
            if self.need_feature_loss:
                # forward F
                # self.real_fea: leaf, not requires_grad (gt features, do not need bp)
                real_fea = self.netF(self.real_H).detach()
                # self.fake_fea: not leaf, requires_grad (need bp, in the graph)
                # self.real_fea and self.fake_fea are not the same, since features is independent to conv
                fake_fea = self.netF(self.fake_H)
                loss_g_fea = self.loss_feature_weight * self.criterion_feature(fake_fea, real_fea)
            # forward D
            pred_g_fake = self.netD(self.fake_H)
            loss_g_gan = self.loss_gan_weight * self.criterion_gan(pred_g_fake, True)

            # total los
            if self.need_pixel_loss:
                if self.need_feature_loss:
                    loss_g_total = loss_g_pixel + loss_g_fea + loss_g_gan
                else:
                    loss_g_total = loss_g_pixel + loss_g_gan
            else:
                loss_g_total = loss_g_fea + loss_g_gan
            loss_g_total.backward()
            self.optimizer_G.step()

        # D
        self.optimizer_D.zero_grad()
        # real data
        pred_d_real = self.netD(self.real_ref)
        loss_d_real = self.criterion_gan(pred_d_real, True)
        # fake data
        pred_d_fake = self.netD(self.fake_H.detach()) # detach to avoid BP to G
        loss_d_fake = self.criterion_gan(pred_d_fake, False)
        if self.opt['train']['gan_type'] == 'wgan-gp':
            n = self.real_ref.size(0)
            if not self.random_pt.size(0) == n:
                self.random_pt.data.resize_(n, 1, 1, 1)
            self.random_pt.data.uniform_()  # Draw random interpolation points
            interp = (self.random_pt * self.fake_H + (1 - self.random_pt) * self.real_ref).detach()
            interp.requires_grad = True
            interp_crit = self.netD(interp)
            loss_d_gp = self.loss_gp_weight * self.criterion_gp(interp, interp_crit)
            # total loss
            loss_d_total = loss_d_real + loss_d_fake + loss_d_gp
        else:
            # total loss
            loss_d_total = loss_d_real + loss_d_fake
        loss_d_total.backward()
        self.optimizer_D.step()

        # set D outputs
        self.Dout_dict = OrderedDict()
        self.Dout_dict['D_out_real'] = torch.mean(pred_d_real.data)
        self.Dout_dict['D_out_fake'] = torch.mean(pred_d_fake.data)

        # set losses
        self.loss_dict = OrderedDict()
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            self.loss_dict['loss_g_pixel'] = loss_g_pixel.data[0] if self.need_pixel_loss else -1
            self.loss_dict['loss_g_fea'] = loss_g_fea.data[0] if self.need_feature_loss else -1
            self.loss_dict['loss_g_gan'] = loss_g_gan.data[0]
        self.loss_dict['loss_d_real'] = loss_d_real.data[0]
        self.loss_dict['loss_d_fake'] = loss_d_fake.data[0]
        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.loss_dict['loss_d_gp'] = loss_d_gp.data[0]

    def val(self):
        self.fake_H = self.netG(self.real_L)

    def test(self):
        self.fake_H = self.netG(self.real_L)

    def get_current_losses(self):
        return self.loss_dict

    def get_more_training_info(self):
        return self.Dout_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.real_L.data[0]
        out_dict['SR'] = self.fake_H.data[0]
        if need_HR:
            out_dict['HR'] = self.real_H.data[0]
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_decsription(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

            # Discriminator
            s, n = self.get_network_decsription(self.netD)
            print('Number of parameters in D: {:,d}'.format(n))
            message = '\n\n\n-------------- Discriminator --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

            if self.need_feature_loss:
                # Perceptual Features
                s, n = self.get_network_decsription(self.netF)
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
                with open(network_path, 'a') as f:
                    f.write(message)

    def load(self):
        if self.load_path_G is not None:
            print('loading model for G [%s] ...' % self.load_path_G)
            self.load_network(self.load_path_G, self.netG)
        if self.opt['is_train'] and self.load_path_D is not None:
            print('loading model for D [%s] ...' % self.load_path_D)
            self.load_network(self.load_path_D, self.netD)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)

    def train(self):
        self.netG.train()
        self.netD.train()

    def eval(self):
        self.netG.eval()
        if self.opt['is_train']:
            self.netD.eval()
