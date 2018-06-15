import torch

pretrained_net = torch.load('../../experiments/pretrained_models/SRResNet_bicx2_in3nf64nb16.pth')
# should run train debug mode first to get an initial model
crt_net = torch.load('../../experiments/debug_SRResNet_bicx4_in3nf64nb16/models/8_G.pth')

for k, v in crt_net.items():
    print(k)
for k, v in crt_net.items():
    if k in pretrained_net:
        crt_net[k] = pretrained_net[k]
        print('replace ... ', k)

# x2 -> x4
crt_net['model.5.weight'] = pretrained_net['model.2.weight']
crt_net['model.5.bias'] = pretrained_net['model.2.bias']
crt_net['model.8.weight'] = pretrained_net['model.5.weight']
crt_net['model.8.bias'] = pretrained_net['model.5.bias']
crt_net['model.10.weight'] = pretrained_net['model.7.weight']
crt_net['model.10.bias'] = pretrained_net['model.7.bias']
torch.save(crt_net, '../pretrained_tmp.pth')

# x2 -> x3
'''
in_filter = pretrained_net['model.2.weight'] # 256, 64, 3, 3
new_filter = torch.Tensor(576, 64, 3, 3)
new_filter[0:256, :, :, :] = in_filter
new_filter[256:512, :, :, :] = in_filter
new_filter[512:, :, :, :] = in_filter[0:576-512, :, :, :]
crt_net['model.2.weight'] = new_filter

in_bias = pretrained_net['model.2.bias']  # 256, 64, 3, 3
new_bias = torch.Tensor(576)
new_bias[0:256] = in_bias
new_bias[256:512] = in_bias
new_bias[512:] = in_bias[0:576 - 512]
crt_net['model.2.bias'] = new_bias

torch.save(crt_net, '../pretrained_tmp.pth')
'''

# x2 -> x8
'''
crt_net['model.5.weight'] = pretrained_net['model.2.weight']
crt_net['model.5.bias'] = pretrained_net['model.2.bias']
crt_net['model.8.weight'] = pretrained_net['model.2.weight']
crt_net['model.8.bias'] = pretrained_net['model.2.bias']
crt_net['model.11.weight'] = pretrained_net['model.5.weight']
crt_net['model.11.bias'] = pretrained_net['model.5.bias']
crt_net['model.13.weight'] = pretrained_net['model.7.weight']
crt_net['model.13.bias'] = pretrained_net['model.7.bias']
torch.save(crt_net, '../pretrained_tmp.pth')
'''

# x3/4/8 RGB -> Y
'''

in_filter = pretrained_net['model.0.weight']
in_new_filter = in_filter[:,0,:,:]*0.2989 + in_filter[:,1,:,:]*0.587 + in_filter[:,2,:,:]*0.114
in_new_filter.unsqueeze_(1)
crt_net['model.0.weight'] = in_new_filter

out_filter = pretrained_net['model.13.weight']
out_new_filter = out_filter[0, :, :, :] * 0.2989 + out_filter[1, :, :, :] * 0.587 + \
    out_filter[2, :, :, :] * 0.114
out_new_filter.unsqueeze_(0)
crt_net['model.13.weight'] = out_new_filter
out_bias = pretrained_net['model.13.bias']
out_new_bias = out_bias[0] * 0.2989 + out_bias[1] * 0.587 + out_bias[2] * 0.114
out_new_bias = torch.Tensor(1).fill_(out_new_bias)
crt_net['model.13.bias'] = out_new_bias
torch.save(crt_net, '../pretrained_tmp.pth')
'''
