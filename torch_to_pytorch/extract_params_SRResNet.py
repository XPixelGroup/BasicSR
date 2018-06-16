import torch
from torch.utils.serialization import load_lua


# # SRResNet w/o BN
# net = load_lua('torch_models/04_SRResNet_noBN_nnCPU.t7')
# py_model = torch.load('pytorch_models/SRResNet_noBN.pth')

# for k, v in py_model.items():
#     print(k)
# ori_keys = len(py_model)
# # first conv
# py_model['model.0.weight'] = net.modules[0].weight
# py_model['model.0.bias'] = net.modules[0].bias
# # 16 Residual Blocks
# for i in range(16):
#     t_conv1_w = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[1].weight
#     t_conv1_b = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[1].bias
#     t_conv2_w = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[3].weight
#     t_conv2_b = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[3].bias

#     py_model['model.1.sub.' + str(i) + '.res.1.weight'] = t_conv1_w
#     py_model['model.1.sub.' + str(i) + '.res.1.bias'] = t_conv1_b
#     py_model['model.1.sub.' + str(i) + '.res.3.weight'] = t_conv2_w
#     py_model['model.1.sub.' + str(i) + '.res.3.bias'] = t_conv2_b
# # LR_conv
# py_model['model.1.sub.16.weight'] = net.modules[1].modules[0].modules[16].weight
# py_model['model.1.sub.16.bias'] = net.modules[1].modules[0].modules[16].bias
# # Up_conv1
# py_model['model.3.weight'] = net.modules[4].weight
# py_model['model.3.bias'] = net.modules[4].bias
# # Up_conv2
# py_model['model.6.weight'] = net.modules[7].weight
# py_model['model.6.bias'] = net.modules[7].bias
# # HR_conv1
# py_model['model.8.weight'] = net.modules[9].weight
# py_model['model.8.bias'] = net.modules[9].bias
# # final_conv
# py_model['model.10.weight'] = net.modules[11].weight
# py_model['model.10.bias'] = net.modules[11].bias

# assert ori_keys == len(py_model), 'the keys mismatch! Please check.'

# torch.save(py_model, 'SRResNet_noBN_torch.pth')


# SRResNet w/ BN
net = load_lua('torch_models/01_SRResNet_nnCPU.t7')
py_model = torch.load('pytorch_models/SRResNet.pth')

for k, v in py_model.items():
    print(k)
ori_keys = len(py_model)
# first conv
py_model['model.0.weight'] = net.modules[0].weight
py_model['model.0.bias'] = net.modules[0].bias
# 16 Residual Blocks
for i in range(16):
    t_bn1_w = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[0].weight
    t_bn1_b = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[0].bias
    t_bn1_rm = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[0].running_mean
    t_bn1_rv = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[0].running_var

    t_conv1_w = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[2].weight
    t_conv1_b = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[2].bias

    t_bn2_w = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[3].weight
    t_bn2_b = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[3].bias
    t_bn2_rm = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[3].running_mean
    t_bn2_rv = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[3].running_var

    t_conv2_w = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[5].weight
    t_conv2_b = net.modules[1].modules[0].modules[i].modules[0].modules[0].modules[5].bias

    py_model['model.1.sub.' + str(i) + '.res.0.weight'] = t_bn1_w
    py_model['model.1.sub.' + str(i) + '.res.0.bias'] = t_bn1_b
    py_model['model.1.sub.' + str(i) + '.res.0.running_mean'] = t_bn1_rm
    py_model['model.1.sub.' + str(i) + '.res.0.running_var'] = t_bn1_rv

    py_model['model.1.sub.' + str(i) + '.res.2.weight'] = t_conv1_w
    py_model['model.1.sub.' + str(i) + '.res.2.bias'] = t_conv1_b

    py_model['model.1.sub.' + str(i) + '.res.3.weight'] = t_bn2_w
    py_model['model.1.sub.' + str(i) + '.res.3.bias'] = t_bn2_b
    py_model['model.1.sub.' + str(i) + '.res.3.running_mean'] = t_bn2_rm
    py_model['model.1.sub.' + str(i) + '.res.3.running_var'] = t_bn2_rv

    py_model['model.1.sub.' + str(i) + '.res.5.weight'] = t_conv2_w
    py_model['model.1.sub.' + str(i) + '.res.5.bias'] = t_conv2_b

# LR bn
py_model['model.1.sub.16.weight'] = net.modules[1].modules[0].modules[16].weight
py_model['model.1.sub.16.bias'] = net.modules[1].modules[0].modules[16].bias
py_model['model.1.sub.16.running_mean'] = net.modules[1].modules[0].modules[16].running_mean
py_model['model.1.sub.16.running_var'] = net.modules[1].modules[0].modules[16].running_var
# LR_conv
py_model['model.1.sub.17.weight'] = net.modules[1].modules[0].modules[17].weight
py_model['model.1.sub.17.bias'] = net.modules[1].modules[0].modules[17].bias
# Up_conv1
py_model['model.3.weight'] = net.modules[4].weight
py_model['model.3.bias'] = net.modules[4].bias
# Up_conv2
py_model['model.6.weight'] = net.modules[7].weight
py_model['model.6.bias'] = net.modules[7].bias
# HR_conv1
py_model['model.8.weight'] = net.modules[9].weight
py_model['model.8.bias'] = net.modules[9].bias
# final_conv
py_model['model.10.weight'] = net.modules[11].weight
py_model['model.10.bias'] = net.modules[11].bias

assert ori_keys == len(py_model), 'the keys mismatch! Please check.'

torch.save(py_model, 'SRResNet_torch.pth')