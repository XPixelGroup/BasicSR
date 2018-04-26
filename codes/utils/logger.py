import os
from datetime import datetime
from utils.util import get_timestamp
# from utils.pavi_logger import PaviLogger


class Logger(object):
    def __init__(self, opt):
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
        # if 'test' not in opt.name:
        #     # pavi logger
        #     url = 'http://pavi.parrotsdnn.org/log'
        #     username = 'NTIRE18'
        #     password = '123455'
        #     self.pavi_logger = PaviLogger(url, username, password=password)
        #     self.pavi_logger.setup(pavi_info)

    def print_format_results(self, mode, rlt):
        epoch = rlt.pop('epoch')
        iters = rlt.pop('iters')
        time = rlt.pop('time')
        model = rlt.pop('model')
        message = '<epoch:{:3d}, iter:{:9,d}, time: {:.2f}> '.format(epoch, iters, time)
        if mode == 'train':
            if model == 'sr':
                loss_pixel = rlt['loss_pixel']  if 'loss_pixel' in rlt else -1
                lr = rlt['lr']
                format_str = '<loss: {:.2e}> lr: {:.2e}'.format(loss_pixel, lr)
            elif model == 'srgan':
                loss_g_pixel = rlt['loss_g_pixel']  if 'loss_g_pixel' in rlt else -1
                loss_g_fea = rlt['loss_g_fea']  if 'loss_g_fea' in rlt else -1
                loss_g_gan = rlt['loss_g_gan']  if 'loss_g_gan' in rlt else -1
                loss_d_real = rlt['loss_d_real']  if 'loss_d_real' in rlt else -1
                loss_d_fake = rlt['loss_d_fake'] if 'loss_d_fake' in rlt else -1
                D_out_real = rlt['D_out_real']  if 'D_out_real' in rlt else -1
                D_out_fake = rlt['D_out_fake']  if 'D_out_fake' in rlt else -1
                lr = rlt['lr']

                if 'loss_d_gp' in rlt:
                    loss_d_gp = rlt['loss_d_gp']
                    format_str = ('<loss_G: pixel: {:.2e}, fea: {:.2e}, gan: {:.2e}><loss_D: '
                        'real: {:.2e} , fake: {:.2e}, gp: {:.2e}><Dout: G: {:.2f}, D: {:.2f}> lr: {:.2e}'.format(\
                        loss_g_pixel, loss_g_fea, loss_g_gan, loss_d_real, loss_d_fake, loss_d_gp, D_out_real, \
                        D_out_fake, lr))
                else:
                    format_str = ('<loss_G: pixel: {:.2e}, fea: {:.2e}, gan: {:.2e}><loss_D: '
                        'real: {:.2e} , fake: {:.2e}><Dout: G: {:.2f}, D: {:.2f}> lr: {:.2e}'.format(\
                        loss_g_pixel, loss_g_fea, loss_g_gan, loss_d_real, loss_d_fake, D_out_real, \
                        D_out_fake, lr))
            message += format_str
        else:
            for label, value in rlt.items():
                message += '%s: %.2e ' % (label, value)
        # print in console
        print(message)
        # write in log file
        if mode == 'train':
            with open(self.loss_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
        elif mode == 'val':
            with open(self.val_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
