import os
import sys
from utils.util import get_timestamp


# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_path, 'print_log.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Logger(object):
    def __init__(self, opt):
        self.exp_name = opt['name']
        self.use_tb_logger = opt['use_tb_logger']
        self.opt = opt['logger']
        self.log_dir = opt['path']['log']
        # loss log file
        self.loss_log_path = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.loss_log_path, "a") as log_file:
            log_file.write('=============== Time: ' + get_timestamp() + ' =============\n')
            log_file.write('================ Training Losses ================\n')
        # val results log file
        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        with open(self.val_log_path, "a") as log_file:
            log_file.write('================ Time: ' + get_timestamp() + ' ===============\n')
            log_file.write('================ Validation Results ================\n')
        if self.use_tb_logger and 'debug' not in self.exp_name:
            from tensorboard_logger import Logger as TensorboardLogger
            self.tb_logger = TensorboardLogger('../tb_logger/' + self.exp_name)

    # def print_format_results(self, mode, rlt):
    #     epoch = rlt.pop('epoch')
    #     iters = rlt.pop('iters')
    #     time = rlt.pop('time')
    #     model = rlt.pop('model')
    #     message = '<epoch:{:3d}, iter:{:9,d}, time: {:.2f}> '.format(epoch, iters, time)
    #     if mode == 'train':
    #         if 'gan' in model: # srgan, sftgan, sftgan_acd
    #             loss_g_pixel = rlt['loss_g_pixel']  if 'loss_g_pixel' in rlt else -1
    #             loss_g_fea = rlt['loss_g_fea']  if 'loss_g_fea' in rlt else -1
    #             loss_g_gan = rlt['loss_g_gan']  if 'loss_g_gan' in rlt else -1
    #             loss_d_real = rlt['loss_d_real']  if 'loss_d_real' in rlt else -1
    #             loss_d_fake = rlt['loss_d_fake'] if 'loss_d_fake' in rlt else -1
    #             D_out_real = rlt['D_out_real']  if 'D_out_real' in rlt else -1
    #             D_out_fake = rlt['D_out_fake']  if 'D_out_fake' in rlt else -1
    #             lr = rlt['lr']

    #             # tensorboard logger - common
    #             if self.use_tb_logger and 'debug' not in self.exp_name:
    #                 if loss_g_pixel != -1 :
    #                     self.tb_logger.log_value('loss_g_pixel', loss_g_pixel, iters)
    #                 if loss_g_fea != -1:
    #                     self.tb_logger.log_value('loss_g_fea', loss_g_fea, iters)
    #                 self.tb_logger.log_value('loss_g_gan', loss_g_gan, iters)
    #                 self.tb_logger.log_value('loss_d_real', loss_d_real, iters)
    #                 self.tb_logger.log_value('loss_d_fake', loss_d_fake, iters)

    #             if 'loss_d_gp' in rlt: # wgan-gp
    #                 loss_d_gp = rlt['loss_d_gp']
    #                 format_str = ('<loss_G: pixel: {:.2e}, fea: {:.2e}, gan: {:.2e}><loss_D: '
    #                     'real: {:.2e} , fake: {:.2e}, gp: {:.2e}><Dout: G: {:.2f}, D: {:.2f}> '
    #                     'lr: {:.2e}'.format(loss_g_pixel, loss_g_fea, loss_g_gan, loss_d_real, \
    #                     loss_d_fake, loss_d_gp, D_out_real, D_out_fake, lr))
    #                 # tensorboard logger - wgan-gp
    #                 if self.use_tb_logger and 'debug' not in self.exp_name:
    #                     self.tb_logger.log_value('loss_d_gp', loss_d_gp, iters)
    #                     self.tb_logger.log_value('Wasserstein_dist', D_out_real - D_out_fake, iters)

    #             else:
    #                 format_str = ('<loss_G: pixel: {:.2e}, fea: {:.2e}, gan: {:.2e}><loss_D: '
    #                     'real: {:.2e} , fake: {:.2e}><Dout: G: {:.2f}, D: {:.2f}> '
    #                     'lr: {:.2e}'.format(loss_g_pixel, loss_g_fea, loss_g_gan, loss_d_real, \
    #                     loss_d_fake, D_out_real, D_out_fake, lr))

    #                 # tensorboard logger - vanilla gan | lsgan
    #                 if self.use_tb_logger and 'debug' not in self.exp_name:
    #                     self.tb_logger.log_value('D_out_real', D_out_real, iters)
    #                     self.tb_logger.log_value('D_out_fake', D_out_fake, iters)

    #         else: # sr and others
    #             loss_pixel = rlt['loss_pixel']  if 'loss_pixel' in rlt else -1
    #             lr = rlt['lr']
    #             format_str = '<loss: {:.2e}> lr: {:.2e}'.format(loss_pixel, lr)
    #             # tensorboard logger
    #             if self.use_tb_logger and 'debug' not in self.exp_name:
    #                 self.tb_logger.log_value('loss_pixel', loss_pixel, iters)

    #         message += format_str
    #     else:
    #         for label, value in rlt.items():
    #             message += '%s: %.4e ' % (label, value)
    #             # tensorboard logger
    #             if self.use_tb_logger and 'debug' not in self.exp_name:
    #                 self.tb_logger.log_value(label, value, iters)

    #     # print in console
    #     print(message)
    #     # write in log file
    #     if mode == 'train':
    #         with open(self.loss_log_path, "a") as log_file:
    #             log_file.write('%s\n' % message)
    #     elif mode == 'val':
    #         with open(self.val_log_path, "a") as log_file:
    #             log_file.write('%s\n' % message)

    def print_format_results(self, mode, rlt):
        epoch = rlt.pop('epoch')
        iters = rlt.pop('iters')
        time = rlt.pop('time')
        model = rlt.pop('model')
        if 'lr' in rlt:
            lr = rlt.pop('lr')
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}, lr:{:.1e}> '.format(
                epoch, iters, time, lr)
        else:
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}> '.format(epoch, iters, time)

        for label, value in rlt.items():
            message += '%s: %.2e ' % (label, value)
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                self.tb_logger.log_value(label, value, iters)

        # print in console
        print(message)
        # write in log file
        if mode == 'train':
            with open(self.loss_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
        elif mode == 'val':
            with open(self.val_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
