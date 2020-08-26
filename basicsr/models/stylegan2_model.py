import math
import mmcv
import random
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from torch import autograd
from torch.nn import functional as F

from basicsr.models import networks as networks
from basicsr.models.base_model import BaseModel
from basicsr.utils import tensor2img


class StyleGAN2Model(BaseModel):
    """StyleGAN2 model."""

    def __init__(self, opt):
        super(StyleGAN2Model, self).__init__(opt)

        # define network net_g
        self.net_g = networks.define_net_g(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'], param_key)

        # latent dimension: self.num_style_feat
        self.num_style_feat = opt['network_g']['num_style_feat']
        self.fixed_sample = torch.randn(
            16, self.num_style_feat, device=self.device)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        # define network net_d
        self.net_d = networks.define_net_d(deepcopy(self.opt['network_d']))
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_model_d', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path,
                              self.opt['path']['strict_load'])

        # define network net_g with Exponential Moving Average (EMA)
        # net_g_ema only used for testing on one GPU and saving, do not need to
        # wrap with DistributedDataParallel
        self.net_g_ema = networks.define_net_g(
            deepcopy(self.opt['network_g'])).to(self.device)
        # load pretrained model
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g_ema, load_path,
                              self.opt['path']['strict_load'], 'params_ema')
        else:
            self.model_ema(0)  # copy net_g weight

        self.net_g.train()
        self.net_d.train()
        self.net_g_ema.eval()

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        self.net_g_reg_every = train_opt['net_g_reg_every']
        self.net_d_reg_every = train_opt['net_d_reg_every']
        self.mixing_prob = train_opt['mixing_prob']

        self.mean_path_length = 0

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        net_g_reg_ratio = self.net_g_reg_every / (self.net_g_reg_every + 1)
        if self.opt['network_g']['type'] == 'StyleGAN2GeneratorC':
            normal_params = []
            style_mlp_params = []
            modulation_conv_params = []
            for name, param in self.net_g.named_parameters():
                if 'modulation' in name:
                    normal_params.append(param)
                elif 'style_mlp' in name:
                    style_mlp_params.append(param)
                elif 'modulated_conv' in name:
                    modulation_conv_params.append(param)
                else:
                    normal_params.append(param)
            optim_params_g = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': style_mlp_params,
                    'lr': train_opt['optim_g']['lr'] * 0.01
                },
                {
                    'params': modulation_conv_params,
                    'lr': train_opt['optim_g']['lr'] / 3
                }
            ]
        else:
            normal_params = []
            for name, param in self.net_g.named_parameters():
                normal_params.append(param)
            optim_params_g = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_g']['lr']
            }]

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                optim_params_g,
                lr=train_opt['optim_g']['lr'] * net_g_reg_ratio,
                betas=(0**net_g_reg_ratio, 0.99**net_g_reg_ratio))
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        net_d_reg_ratio = self.net_d_reg_every / (self.net_d_reg_every + 1)
        if self.opt['network_d']['type'] == 'StyleGAN2DiscriminatorC':
            normal_params = []
            linear_params = []
            for name, param in self.net_d.named_parameters():
                if 'final_linear' in name:
                    linear_params.append(param)
                else:
                    normal_params.append(param)
            optim_params_d = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_d']['lr']
                },
                {
                    'params': linear_params,
                    'lr': train_opt['optim_d']['lr'] * (1 / math.sqrt(512))
                }
            ]
        else:
            normal_params = []
            for name, param in self.net_g.named_parameters():
                normal_params.append(param)
            optim_params_d = [{  # add normal params first
                'params': normal_params,
                'lr': train_opt['optim_d']['lr']
            }]

        optim_type = train_opt['optim_d'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_d = torch.optim.Adam(
                optim_params_d,
                lr=train_opt['optim_d']['lr'] * net_d_reg_ratio,
                betas=(0**net_d_reg_ratio, 0.99**net_d_reg_ratio))
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_d)

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(
                net_g_params[k].data, alpha=1 - decay)

    def feed_data(self, data):
        self.real_img = data['gt'].to(self.device)

    def make_noise(self, batch, num_noise):
        if num_noise == 1:
            noises = torch.randn(
                batch, self.num_style_feat, device=self.device)
        else:
            noises = torch.randn(
                num_noise, batch, self.num_style_feat,
                device=self.device).unbind(0)
        return noises

    def mixing_noise(self, batch, prob):
        if random.random() < prob:
            return self.make_noise(batch, 2)
        else:
            return [self.make_noise(batch, 1)]

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True
        self.optimizer_d.zero_grad()

        batch = self.real_img.size(0)
        noise = self.mixing_noise(batch, self.mixing_prob)
        fake_img, _ = self.net_g(noise)
        fake_pred = self.net_d(fake_img.detach())

        real_pred = self.net_d(self.real_img)
        l_d = self.d_logistic_loss(real_pred, fake_pred)
        loss_dict['l_d'] = l_d
        # In WGAN, real_score should be positive and fake_score should be
        # negative
        loss_dict['real_score'] = real_pred.detach().mean()
        loss_dict['fake_score'] = fake_pred.detach().mean()
        l_d.backward()

        if current_iter % self.net_d_reg_every == 0:
            self.real_img.requires_grad = True
            real_pred = self.net_d(self.real_img)
            l_d_r1 = self.d_r1_loss(real_pred, self.real_img)
            l_d_r1 = (
                self.opt['train']['r1_weight'] / 2 * l_d_r1 *
                self.net_d_reg_every + 0 * real_pred[0])
            # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
            # error will arise: RuntimeError: Expected to have finished
            # reduction in the prior iteration before starting a new one.
            # This error indicates that your module has parameters that were
            # not used in producing loss.
            loss_dict['l_d_r1'] = l_d_r1.detach().mean()
            l_d_r1.backward()

        self.optimizer_d.step()

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False
        self.optimizer_g.zero_grad()

        noise = self.mixing_noise(batch, self.mixing_prob)
        fake_img, _ = self.net_g(noise)
        fake_pred = self.net_d(fake_img)

        l_g = self.g_nonsaturating_loss(fake_pred)
        loss_dict['l_g'] = l_g
        l_g.backward()

        if current_iter % self.net_g_reg_every == 0:
            path_batch_size = max(
                1, batch // self.opt['train']['path_batch_shrink'])
            noise = self.mixing_noise(path_batch_size, self.mixing_prob)
            fake_img, latents = self.net_g(noise, return_latents=True)
            l_g_path, path_lengths = self.g_path_regularize(fake_img, latents)

            l_g_path = (
                self.opt['train']['path_weight'] * self.net_g_reg_every *
                l_g_path + 0 * fake_img[0, 0, 0, 0])
            # TODO:  why do we need to add 0 * fake_img[0, 0, 0, 0]
            l_g_path.backward()
            loss_dict['l_g_path'] = l_g_path.detach().mean()
            loss_dict['path_length'] = path_lengths

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # EMA
        self.model_ema(decay=0.5**(32 / (10 * 1000)))

    def d_logistic_loss(self, real_pred, fake_pred):
        """Logistic loss for discriminator. Actually the WGAN loss"""
        # softplus is a smooth approximation to the ReLU function
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred, real_img):
        """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
        grad_real = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0],
                                             -1).sum(1).mean()
        return grad_penalty

    def g_nonsaturating_loss(self, fake_pred):
        """Non-saturating loss for generator. Like d_logistic_loss"""
        loss = F.softplus(-fake_pred).mean()
        return loss

    def g_path_regularize(self, fake_img, latents, decay=0.01):
        noise = torch.randn_like(fake_img) / math.sqrt(
            fake_img.shape[2] * fake_img.shape[3])
        grad = autograd.grad(
            outputs=(fake_img * noise).sum(),
            inputs=latents,
            create_graph=True)[0]
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

        path_mean = self.mean_path_length + decay * (
            path_lengths.mean() - self.mean_path_length)
        self.mean_path_length = path_mean.detach()

        path_penalty = (path_lengths - path_mean).pow(2).mean()

        return path_penalty, path_lengths.detach().mean()

    def test(self):
        with torch.no_grad():
            self.net_g_ema.eval()
            self.output, _ = self.net_g_ema([self.fixed_sample])

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger,
                                    save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        assert dataloader is None, 'Validation dataloader should be None.'
        self.test()
        result = tensor2img(self.output, min_max=(-1, 1))
        if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'],
                                     'train', f'train_{current_iter}.png')
        else:
            save_img_path = osp.join(self.opt['path']['visualization'], 'test',
                                     f'test_{self.opt["name"]}.png')
        mmcv.imwrite(result, save_img_path)
        # add sample images to tb_logger
        result = mmcv.bgr2rgb(result / 255.)
        if tb_logger is not None:
            tb_logger.add_image(
                'samples', result, global_step=current_iter, dataformats='HWC')

    def save(self, epoch, current_iter):
        self.save_network([self.net_g, self.net_g_ema],
                          'net_g',
                          current_iter,
                          param_key=['params', 'params_ema'])
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
