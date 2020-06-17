import importlib
from collections import OrderedDict

import torch

import basicsr.models.networks as networks
from basicsr.models.video_base_model import VideoBaseModel

loss_module = importlib.import_module('basicsr.models.losses')


class VideoGANModel(VideoBaseModel):
    """ Video GAN model.
    """

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = networks.define_net_d(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt', None):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt', None):
            gan_type = train_opt['gan_opt'].pop('type')
            cri_gan_cls = getattr(loss_module, gan_type)
            self.cri_gan = cri_gan_cls(**train_opt['gan_opt']).to(self.device)

        self.net_d_steps = train_opt['net_d_steps'] if train_opt[
            'net_d_steps'] else 1
        self.net_d_init_steps = train_opt['net_d_init_steps'] if train_opt[
            'net_d_init_steps'] else 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        self.log_dict = OrderedDict()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(),
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(self.net_d.parameters(),
                                                **train_opt['optim_d'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, step):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        if step % self.net_d_steps == 0 and step > self.net_d_init_steps:
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                self.log_dict['l_g_pix'] = l_g_pix.item()
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    self.log_dict['l_g_percep'] = l_g_percep.item()
                if l_g_style is not None:
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            self.log_dict['l_g_gan'] = l_g_gan.item()

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # forward and backward separately, since batch norm statistics differ
        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
