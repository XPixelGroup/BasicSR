import torch
from torch.utils.serialization import load_lua


def load_conv(pytorch, pth_name, torch, t7_name):
    pytorch[pth_name + '.weight'] = torch[t7_name].weight
    pytorch[pth_name + '.bias'] = torch[t7_name].bias

# # condition network
# cond_net = load_lua('./models/condition_net_layers.t7')
# py_cond = torch.load('cond_raw.pth')

# for k, v in py_cond.items():
#     print(k)
# ori_keys = len(py_cond)

# load_conv(py_cond, 'CondNet.0', cond_net, 10)
# load_conv(py_cond, 'CondNet.2', cond_net, 8)
# load_conv(py_cond, 'CondNet.4', cond_net, 6)
# load_conv(py_cond, 'CondNet.6', cond_net, 4)
# load_conv(py_cond, 'CondNet.8', cond_net, 1)

# assert ori_keys == len(py_cond), 'the keys mismatch! Please check.'

# torch.save(py_cond, 'condition_net.pth')

# sft network
sft_net = load_lua('./models/sft_net_layers.t7')
py_sft = torch.load('sft_raw.pth')

for k, v in py_sft.items():
    print(k)

for k, v in sft_net.items():
    print(k, v)
ori_keys = len(py_sft)

def load_sft(pytorch, torch, sft_name, scale_idx, shift_idx):
    # scale
    pytorch[sft_name + '.SFT_scale_conv0.weight'] = torch[scale_idx].modules[0].modules[1].modules[0].weight
    pytorch[sft_name + '.SFT_scale_conv0.bias'] = torch[scale_idx].modules[0].modules[1].modules[0].bias
    pytorch[sft_name + '.SFT_scale_conv1.weight'] = torch[scale_idx].modules[0].modules[1].modules[2].weight
    pytorch[sft_name + '.SFT_scale_conv1.bias'] = torch[scale_idx].modules[0].modules[1].modules[2].bias
    # shift
    pytorch[sft_name + '.SFT_shift_conv0.weight'] = torch[shift_idx].modules[0].modules[1].modules[0].weight
    pytorch[sft_name + '.SFT_shift_conv0.bias'] = torch[shift_idx].modules[0].modules[1].modules[0].bias
    pytorch[sft_name + '.SFT_shift_conv1.weight'] = torch[shift_idx].modules[0].modules[1].modules[2].weight
    pytorch[sft_name + '.SFT_shift_conv1.bias'] = torch[shift_idx].modules[0].modules[1].modules[2].bias

def load_resblock_sft(pytorch, torch, blk_name, idxs):
    # idxs = [sft0_scale, sft0_shift, sft1_scale, sft1_shift, conv0_idx, conv1_idx]
    # sft0
    load_sft(pytorch, torch, blk_name + '.sft0', idxs[0], idxs[1])
    # conv0
    load_conv(pytorch, blk_name + '.conv0', torch, idxs[4])
    # sft1
    load_sft(pytorch, torch, blk_name + '.sft1', idxs[2], idxs[3])
    # conv1
    load_conv(pytorch, blk_name + '.conv1', torch, idxs[5])


def load_conv_HRbranch(pytorch, pth_name, torch, t7_name):
    pytorch[pth_name + '.weight'] = t7_name.weight
    pytorch[pth_name + '.bias'] = t7_name.bias


load_conv(py_sft, 'conv0', sft_net, 4)
load_resblock_sft(py_sft, sft_net, 'sft_branch.0', [157, 155, 143, 137, 148, 122])
load_resblock_sft(py_sft, sft_net, 'sft_branch.1', [156, 153, 138, 131, 144, 114])
load_resblock_sft(py_sft, sft_net, 'sft_branch.2', [154, 150, 132, 124, 139, 105])
load_resblock_sft(py_sft, sft_net, 'sft_branch.3', [151, 146, 125, 116, 133, 96])
load_resblock_sft(py_sft, sft_net, 'sft_branch.4', [147, 141, 117, 107, 126, 87])
load_resblock_sft(py_sft, sft_net, 'sft_branch.5', [142, 135, 108, 98, 118, 78])
load_resblock_sft(py_sft, sft_net, 'sft_branch.6', [136, 128, 99, 89, 109, 69])
load_resblock_sft(py_sft, sft_net, 'sft_branch.7', [129, 120, 90, 80, 100, 60])
load_resblock_sft(py_sft, sft_net, 'sft_branch.8', [121, 111, 81, 71, 91, 51])
load_resblock_sft(py_sft, sft_net, 'sft_branch.9', [112, 102, 72, 62, 82, 43])
load_resblock_sft(py_sft, sft_net, 'sft_branch.10', [103, 93, 63, 53, 73, 36])
load_resblock_sft(py_sft, sft_net, 'sft_branch.11', [94, 84, 54, 45, 64, 30])
load_resblock_sft(py_sft, sft_net, 'sft_branch.12', [85, 75, 46, 38, 55, 25])
load_resblock_sft(py_sft, sft_net, 'sft_branch.13', [76, 66, 39, 32, 47, 21])
load_resblock_sft(py_sft, sft_net, 'sft_branch.14', [67, 57, 33, 27, 40, 18])
load_resblock_sft(py_sft, sft_net, 'sft_branch.15', [58, 49, 28, 23, 34, 16])
load_sft(py_sft, sft_net, 'sft_branch.16', 9, 7)
load_conv(py_sft, 'sft_branch.17', sft_net, 5)
# HR branch
load_conv_HRbranch(py_sft, 'HR_branch.1', sft_net, sft_net[1].modules[1])
load_conv_HRbranch(py_sft, 'HR_branch.4', sft_net, sft_net[1].modules[4])
load_conv_HRbranch(py_sft, 'HR_branch.6', sft_net, sft_net[1].modules[6])
load_conv_HRbranch(py_sft, 'HR_branch.8', sft_net, sft_net[1].modules[8])


assert ori_keys == len(py_sft), 'the keys mismatch! Please check.'

print('saving sft_net...')
torch.save(py_sft, 'sft_net.pth')
