import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel


class SRModel(BaseModel):
    def name(self):
        return 'SRModel'

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train']
        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        # define network and load pretrained models
        self.netG = networks.define_G(opt)
        self.load()

        if self.is_train:
            self.netG.train()

            # define loss function
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.criterion_pixel = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pixel = nn.MSELoss()
            else:
                raise NotImplementedError('Loss type [%s] is not recognized.' % loss_type)
            if self.use_gpu:
                self.criterion_pixel.cuda()
            self.loss_pixel_weight = train_opt['pixel_weight']

            # initialize optimizers
            self.optimizers = []
            self.lr_G = train_opt['lr_G']
            self.wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            # can optimize for a part of the model
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [%s] will not optimize.' % k)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=self.lr_G, weight_decay=self.wd_G)
            self.optimizers.append(self.optimizer_G)

            # initialize schedulers
            self.schedulers = []
            if train_opt['lr_scheme'] == 'MultiStepLR':
                self.scheduler_G = lr_scheduler.MultiStepLR(self.optimizer_G, \
                    train_opt['lr_steps'], train_opt['lr_gamma'])
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
            self.schedulers.append(self.scheduler_G)

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, volatile=False, need_HR=True):
        # LR
        input_L = data['LR']
        self.input_L.resize_(input_L.size()).copy_(input_L)
        self.real_L = Variable(self.input_L, volatile=volatile)

        if need_HR:
            input_H = data['HR']
            self.input_H.resize_(input_H.size()).copy_(input_H)
            self.real_H = Variable(self.input_H, volatile=volatile)

        # import torchvision.utils
        # torchvision.utils.save_image(input_L, 'LR.png', nrow=4, padding=2, normalize=False)
        # torchvision.utils.save_image(input_H, 'HR.png', nrow=4, padding=2, normalize=False)

    def forward_G(self):
        self.fake_H = self.netG(self.real_L)

    def backward_G(self):
        self.loss_pixel = self.loss_pixel_weight * self.criterion_pixel(self.fake_H, self.real_H)
        self.loss_pixel.backward()

    def optimize_parameters(self, step):
        self.forward_G()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def test(self):
        self.netG.eval()
        self.fake_H = self.netG(self.real_L)
        self.netG.train()

    def get_current_losses(self):
        out_dict = OrderedDict()
        out_dict['loss_pixel'] = self.loss_pixel.data[0]
        return out_dict

    def get_more_training_info(self):
        return None

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.real_L.data[0].float().cpu()
        out_dict['SR'] = self.fake_H.data[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.data[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_decsription(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [%s] ...' % load_path_G)
            self.load_network(load_path_G, self.netG)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
