import os
import torch
from torch import nn
from torch.autograd import Function

BASICSR_JIT = os.getenv('BASICSR_JIT')
if BASICSR_JIT == 'True':
    from torch.utils.cpp_extension import load
    module_path = os.path.dirname(__file__)
    meta_upscale_ext = load(
        'meta_upscale',
        sources=[
            os.path.join(module_path, 'src', 'meta_upscale.cpp'),
            os.path.join(module_path, 'src', 'meta_upscale_kernel.cu'),
        ],
        # extra_cuda_cflags=['TORCH_USE_CUDA_DSA']
    )
else:
    try:
        from . import meta_upscale_ext
    except ImportError:
        pass
        # avoid annoying print output
        # print(f'Cannot import deform_conv_ext. Error: {error}. You may need to: \n '
        #       '1. compile with BASICSR_EXT=True. or\n '
        #       '2. set BASICSR_JIT=True during running')
        

class MetaUpscaleFunction(Function):
    
    @staticmethod
    def forward(ctx, x, weight, s_v, s_h, batch_size, 
        in_channels, in_height, in_width, 
        out_channels, out_height, out_width, kernel_size):
        x = x.float()
        weight = weight.float()
        out = meta_upscale_ext.forward(x.contiguous(), weight.contiguous(), s_v, s_h,
            batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size)
        
        ctx.save_for_backward(x, weight)
        ctx.s_v, ctx.s_h = s_v, s_h
        ctx.batch_size, ctx.in_channels, ctx.in_height, ctx.in_width = batch_size, in_channels, in_height, in_width
        ctx.out_channels, ctx.out_height, ctx.out_width = out_channels, out_height, out_width
        ctx.kernel_size = kernel_size
        
        return out
    
    @staticmethod
    def backward(ctx, g_out):
        x, weight = ctx.saved_tensors
        s_v, s_h = ctx.s_v, ctx.s_h
        batch_size, in_channels, in_height, in_width = ctx.batch_size, ctx.in_channels, ctx.in_height, ctx.in_width
        out_channels, out_height, out_width = ctx.out_channels, ctx.out_height, ctx.out_width
        kernel_size = ctx.kernel_size
        
        dx, dweight = meta_upscale_ext.backward(g_out.contiguous(), x.contiguous(), weight.contiguous(), s_v, s_h,
            batch_size, in_channels, in_height, in_width, out_channels, out_height, out_width, kernel_size)
        return dx, dweight, None, None, None, None, None, None, None, None, None, None
    
    
class MetaUpscale(nn.Module):

    def __init__(self, s_v, s_h, in_channels, out_channels, kernel_size):
        super().__init__()
        self.s_v = s_v
        self.s_h = s_h
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

    def forward(self, x, weight):
        batch_size = x.size(0)
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = int(self.s_v * in_height)
        out_width = int(self.s_h * in_width)
        # print('in_height: ', in_height)
        # print('in_width: ', in_width)
        # print('out_height: ', out_height)
        # print('out_width: ', out_width)
        x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        return meta_upscale(x, weight, self.s_v, self.s_h, batch_size, self.in_channels, in_height, in_width, 
            self.out_channels, out_height, out_width, self.kernel_size)


def meta_upscale(x, weight, s_v, s_h, batch_size, in_channels, in_height, in_width, 
        out_channels, out_height, out_width, kernel_size):
    return MetaUpscaleFunction.apply(x, weight, s_v, s_h, batch_size, in_channels, in_height, in_width, 
        out_channels, out_height, out_width, kernel_size)
    

# # TEST
# import numpy as np
# from tqdm import tqdm
# from copy import deepcopy
# def meta_upscale_im2col(x, weight, s_v, s_h, batch_size, in_c, out_c, out_h, out_w, kernel_size):
#     n = x.size(0)
#     v_idxes = np.arange(out_h) // s_v
#     h_idxes = np.arange(out_w) // s_h
#     x = x[:, :, v_idxes, :][:, :, :, h_idxes] # (n, in_c, out_h, out_w)
    
    
    
#     x = nn.functional.unfold(x, kernel_size, padding=1).permute(2, 0, 1) # (h_out * w_out, n, c_in * k * k)
    
#     # print('x_up: ', x.shape)
#     # print(x)
#     # for i, j in zip((3, 3, 4, 4), (3, 4, 3, 4)):
#     #     i_p = int(i / s_v)
#     #     j_p = int(j / s_h)
#     #     print('i, j, i_p, j_p, x:', i, j, i_p, j_p, x[j + i * out_w, 0, 0])
    
#     weight = weight.view(out_h * out_w, kernel_size * kernel_size, in_c, out_c).permute(0, 2, 1, 3)
#     weight = weight.contiguous().view(out_h * out_w, c_in * kernel_size * kernel_size, out_c)
#     # (h_out * w_out, n, c_in * k * k) @ (h_out * w_out, c_in * k * k, out_c) -> (h_out * w_out, n, out_c)
#     out = torch.bmm(x, weight).permute(1, 2, 0)
#     return out.contiguous().view(n, out_c, out_h, out_w)

# def meta_upscale_naive(x, weight, s_v, s_h, batch_size, in_c, out_c, out_h, out_w, kernel_size):
#     # print('x: ', x.shape)
#     # print(x)
#     weight = weight.view(out_h, out_w, kernel_size, kernel_size, in_c, out_c)
#     out = torch.zeros(batch_size, out_c, out_h, out_w, requires_grad=True).cuda()
#     x = nn.functional.pad(x, (1, 1, 1, 1))
#     # print(x.shape)
#     for i in tqdm(range(out_h)):
#         for j in range(out_w):
#             i_p = int(i / s_v)
#             j_p = int(j / s_h)
#             # if i_p == 1 and j_p == 1:
#             #     print('i, j, i_p, j_p, x: ', i, j, i_p, j_p, x[0, 0, 1, 1])
#             for k1 in range(kernel_size):
#                 for k2 in range(kernel_size):
#                     for ci in range(in_c):    
#                         for co in range(out_c):
#                             out[:, co, i, j] += x[:, ci, i_p + k1, j_p + k2] * weight[i, j, k1, k2, ci, co]
#     return out

# if __name__ == '__main__':
#     # h, w = 96, 144
#     # H, W = 721, 1440
#     h, w = 6 * 6, 9 * 6
#     H, W = 60 * 6 + 1, 90 * 6
#     s_v, s_h = H / h, W / w
#     c_in, c_out = 1, 1
#     batch_size = 4
    
#     x = torch.randn(batch_size, c_in, h, w).cuda()
#     x_cuda = x.clone()
#     x_naive = x.clone()
#     x_cuda.requires_grad_(True)
#     x_naive.requires_grad_(True)
    
#     weight = torch.randn(H * W, 3 * 3 * c_in * c_out, requires_grad=True).cuda()
#     w_cuda = weight
#     w_naive = weight
    
    
#     # out_im2col = meta_upscale_im2col(x1, w1, s_v, s_h, batch_size, c_in, c_out, H, W, 3)
#     # print(out_im2col.shape)
#     # print(out_im2col.device)
    
#     up = MetaUpscale(s_v, s_h, batch_size, c_in, h, w, c_out, H, W, 3)
#     out = up(x_cuda, w_cuda)
#     print(out.shape)
#     print(out.device)
    
#     out_naive = meta_upscale_naive(x_naive, w_naive, s_v, s_h, batch_size, c_in, c_out, H, W, 3)
#     print(out_naive.shape)
#     print(out_naive.device)
    
#     # print('naive - im2col: ', torch.norm(out_naive - out_im2col))
#     # print(out_im2col)
#     # print(out_naive)
    
    
    
#     print('naive - cuda: ', torch.norm(out_naive - out))
    
#     gt = torch.randn(batch_size, c_out, H, W).cuda()
#     loss = nn.functional.mse_loss(out, gt)
#     # loss = out.mean()
    
#     # loss_naive = out_naive.mean()
#     loss_naive = nn.functional.mse_loss(out_naive, gt)
    
#     print('loss: ', loss)
#     print('loss_naive: ', loss_naive)
    
#     print('loss backward')
#     loss.backward()
#     print('loss_naive backward')
#     loss_naive.backward()
    
    
#     # print(x_cuda.grad)
#     # print(x_naive.grad)
    
#     print('grad norm: ', torch.norm(x_cuda.grad - x_naive.grad))
    
#     # print(out)
#     # print(out_im2col)