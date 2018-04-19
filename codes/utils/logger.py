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

    def print_results(self, mode, results):
        epoch = results.pop('epoch')
        iters = results.pop('iters')
        time = results.pop('time')
        if mode == 'train':
            lr = results.pop('lr')

        message = '(epoch:{:3d}, iter:{:9,d}, time: {:.3f}) '.format(epoch, iters, time)
        for label, value in results.items():
            message += '%s: %.2e ' % (label, value)
            # if mode == 'loss':
            #     train_loss = value
            # else:
            #     test_PSNR = value
        if mode == 'train':
            message += 'lr: %.2e' % lr
        # print in console
        print(message)
        # write in log file
        if mode == 'train':
            # if 'test' not in self.model_name:
            #     send_data_train = {'loss':train_loss}
            #     log_data = {'time':str(datetime.now()), 'flow_id':'train', 'iter_num':iters, 'outputs':send_data_train}
            #     self.pavi_logger.log(log_data)
            with open(self.loss_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
        elif mode == 'val':
            # if 'test' not in self.model_name:
            #     send_data_test = {'acc_PSNR':test_PSNR}
            #     log_data = {'time':str(datetime.now()), 'flow_id':'test', 'iter_num':iters, 'outputs':send_data_test}
            #     self.pavi_logger.log(log_data)
            with open(self.val_log_path, "a") as log_file:
                log_file.write('%s\n' % message)
