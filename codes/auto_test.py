'''auto test several models.'''

import json
import os

test_json_path = 'options/test/test_esrgan_auto.json'


def modify_json(json_path, model_name, iteration):
    with open(json_path, 'r+') as json_file:
        config = json.load(json_file)

        config['name'] = model_name
        config['datasets']['test_1']['name'] = 'pirm_test_{:d}k'.format(iteration)
        # config['datasets']['test_1']['dataroot_LR'] = \
        #   '/mnt/SSD/xtwang/BasicSR_datasets/PIRM/PIRM_Test_set/LR'
        config['path']['pretrain_model_G'] = \
            '../experiments/{:s}/models/{:d}_G.pth'.format(model_name, iteration*1000)
        json_file.seek(0)  # rewind
        json.dump(config, json_file)
        json_file.truncate()  # if the new data is smaller than the previous


model_iter_dict = {}
model_iter_dict['100_ESRGAN_SRResNet_pristine_pixel10_minc'] = [80, 85, 90, 95]

for model_name, iter_list in model_iter_dict.items():
    for iteration in iter_list:
        modify_json(test_json_path, model_name, iteration)
        # run test scripts
        print('\n\nTesting {:s} {:d}k...'.format(model_name, iteration))
        os.system('python test.py -opt ' + test_json_path)
