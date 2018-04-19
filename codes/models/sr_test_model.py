from collections import OrderedDict
from functools import reduce
import time

import torch
from torch.autograd import Variable

import models.networks as networks
import models.modules.block as B
from .modules.util import load_network
from .base_model import BaseModel

def chop_forward(x, model, scale, shave=10, min_size=80000, n_GPUs=1):
    n_GPUs = min(n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, n_GPUs):
            input_batch = torch.cat(inputlist[i:(i + n_GPUs)], dim=0)
            output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(n_GPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, n_GPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(b, c, h, w), volatile=True)
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


class SRTestModel(BaseModel):
    def name(self):
        return 'SRTestModel'

    def initialize(self, opt):
        super(SRTestModel, self).initialize(opt)
        assert not opt.is_train
        self.netG = networks.define_G(opt).eval()
        self.load_path_G = opt.path.pretrain_model_G
        assert self.load_path_G is not None
        self.load()

        # by default, in half precision
        # will not hurt the final image results and PSNR
        # use less GPU memory largely, and less test time
        if True: # precision == 'half'
            self.Tensor = torch.cuda.HalfTensor if opt.gpu_ids else torch.Tensor
            self.netG.half()
        self.input_L = self.Tensor()
        self.input_H = self.Tensor()

        self.test_time = 0
        self.counter = 0

        print('---------- Model initialized -------------')


    def feed_data(self, data, volatile=True):
        input_H = data['HR']
        input_L = data['LR']
        self.input_H.resize_(input_H.size()).copy_(input_H)
        self.input_L.resize_(input_L.size()).copy_(input_L)
        self.real_H = Variable(self.input_H, volatile=volatile) # in range [0,1]
        self.real_L = Variable(self.input_L, volatile=volatile) # in range [0,1]

    def test(self, mode='normal'):
        self.counter += 1
        start_time = time.time()
        if mode == 'normal':
            self.fake_H = self.netG(self.real_L)
        elif mode == 'enhanced_prediction':
            self.enhanced_prediction(multi_GPU=True)
        elif mode == 'chop_forward':
            self.fake_H = chop_forward(self.real_L, self.netG, 4, shave=10, min_size=80000, n_GPUs=1)
        else:
            raise NotImplementedError('test mode [{}] not implemented.'.format(mode))
        use_time = time.time() - start_time
        if self.counter != 1:
            self.test_time += use_time
            ave_time = self.test_time / (self.counter-1)
        else:
            ave_time = use_time
        print('Time: {:6.2f} sec, average: {:6.2f} sec.'.format(use_time, ave_time))

    def enhanced_prediction(self, multi_GPU=False):
        """Predict images with rotations and flips and then average them.
        (See paper: Seven ways to improve example-based single image super resolution)
        (Codes are modified from https://github.com/thstkdgus35/EDSR-PyTorch)

        Args:
            precision(str): calculation precision, half | single | double

        Returns:
            tensor: averaged predictions
        """
        def _transform(v, op):
            v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'vflip':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'hflip':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 'transpose':
                tfnp = v2np.transpose((0,1,3,2)).copy()
            ret = self.Tensor(tfnp).cuda()
            return Variable(ret, volatile=v.volatile)

        in_list = [self.real_L]
        # rotations and flips
        for tf in 'vflip', 'hflip', 'transpose': # for 8 times
            # for tf in 'hflip', 'transpose':
            in_list.extend([_transform(t, tf) for t in in_list])
        if multi_GPU: # by default, use 4 GPUs
            out_list = []
            for i in range(0, 8, 4): # total 2 batches
                in_batch = torch.cat(in_list[i:i+4], dim=0)
                out_batch = self.netG(in_batch)
                out_list.extend(out_batch.chunk(4, dim=0))
        else:
            out_list = [self.netG(img) for img in in_list]
        # rotate or flip to the original's
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 'transpose')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'hflip')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'vflip')
        self.fake_H = reduce((lambda x, y: x + y), out_list) / len(out_list)



    def val(self):
        self.test()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['low-resolution'] = self.real_L.data[0]
        out_dict['super-resolution'] = self.fake_H.data[0]
        out_dict['ground-truth'] = self.real_H.data[0]
        return out_dict

    def load(self):
        print('loading model for G [%s] ...' % self.load_path_G)
        load_network(self.load_path_G, self.netG)
