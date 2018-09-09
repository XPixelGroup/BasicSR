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
        with open(self.loss_log_path, 'a') as log_file:
            log_file.write('=============== Time: ' + get_timestamp() + ' =============\n')
            log_file.write('================ Training Losses ================\n')
        # val results log file
        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        with open(self.val_log_path, 'a') as log_file:
            log_file.write('================ Time: ' + get_timestamp() + ' ===============\n')
            log_file.write('================ Validation Results ================\n')
        if self.use_tb_logger and 'debug' not in self.exp_name:
            from tensorboard_logger import Logger as TensorboardLogger
            self.tb_logger = TensorboardLogger('../tb_logger/' + self.exp_name)

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
            if mode == 'train':
                message += '{:s}: {:.2e} '.format(label, value)
            elif mode == 'val':
                message += '{:s}: {:.4e} '.format(label, value)
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                self.tb_logger.log_value(label, value, iters)

        # print in console
        print(message)
        # write in log file
        if mode == 'train':
            with open(self.loss_log_path, 'a') as log_file:
                log_file.write(message + '\n')
        elif mode == 'val':
            with open(self.val_log_path, 'a') as log_file:
                log_file.write(message + '\n')
