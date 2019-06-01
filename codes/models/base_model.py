import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
