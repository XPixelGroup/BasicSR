import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
args = parser.parse_args()
options_path = args.opt
opt = option.parse(options_path, is_train=True)

util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
    not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
option.save(opt) # save option file to the opt['path']['options']
opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(opt['path']['log'], 'print_log.txt'), "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = PrintLogger()

# random seed
seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
print("Random Seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)

from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger

def main():
    # create train and val dataloader
    train_loader = None
    val_loader = None
    train_dataset_opt = None
    val_dataset_opt = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_dataset_opt = dataset_opt
            train_set = create_dataset(dataset_opt)
            batch_size = dataset_opt['batch_size']
            train_size = int(math.ceil(len(train_set) / batch_size))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    model.train()

    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # training
            train_start_time = time.time()
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            train_elapsed = time.time() - train_start_time

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                losses = model.get_current_losses()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = train_elapsed
                for k, v in losses.items():
                    print_rlt[k] = v
                more_info = model.get_more_training_info()
                if more_info is not None:
                    for k, v in more_info.items():
                        print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving the model at the end of iter %d' % (current_step))
                model.save(current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                validate(val_loader, model, logger, epoch, current_step, val_dataset_opt)

            # update learning rate
            model.update_learning_rate()

        # print('End of epoch %d.' % epoch)

    print('Saving the final model.')
    model.save('latest')
    print('End of Training \t Time taken: %d sec' % (time.time() - start_time))


def validate(val_loader, model, logger, epoch, current_step, val_dataset_opt):
    print('---------- validation -------------')
    val_start_time = time.time()
    model.eval() # Change to eval mode. It is important for BN layers.

    val_results = OrderedDict()
    avg_psnr = 0.0
    idx = 0
    for val_data in val_loader:
        idx += 1
        img_path = val_data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_dir = os.path.join(opt['path']['val_images'], img_name)
        util.mkdir(img_dir)

        model.feed_data(val_data, volatile=True)
        model.val()

        visuals = model.get_current_visuals()

        sr_img = util.tensor2img_np(visuals['SR']) # uint8
        gt_img = util.tensor2img_np(visuals['HR']) # uint8
        # # modcrop
        # gt_img = util.modcrop(gt_img, val_dataset_opt['scale'])
        h_min = min(sr_img.shape[0], gt_img.shape[0])
        w_min = min(sr_img.shape[1], gt_img.shape[1])
        sr_img = sr_img[0:h_min, 0:w_min, :]
        gt_img = gt_img[0:h_min, 0:w_min, :]

        crop_size = val_dataset_opt['scale'] + 2
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]

        # Save SR images for reference
        save_img_path = os.path.join(img_dir, '%s_%s.png' % (img_name,current_step))
        util.save_img_np(sr_img.squeeze(), save_img_path)

        # TODO need to modify
        # metric_mode = val_dataset_opt['metric_mode']
        # if metric_mode == 'y':
        #     cropped_sr_img = util.rgb2ycbcr(cropped_sr_img, only_y=True)
        #     cropped_gt_img = util.rgb2ycbcr(cropped_gt_img, only_y=True)

        avg_psnr += util.psnr(cropped_sr_img, cropped_gt_img)

    avg_psnr = avg_psnr / idx

    val_elapsed = time.time() - val_start_time
    # Save to log
    print_rlt = OrderedDict()
    print_rlt['model'] = opt['model']
    print_rlt['epoch'] = epoch
    print_rlt['iters'] = current_step
    print_rlt['time'] = val_elapsed
    print_rlt['psnr'] = avg_psnr
    logger.print_format_results('val', print_rlt)
    model.train() # change back to train mode.
    print('-----------------------------------')


if __name__ == '__main__':
    main()
