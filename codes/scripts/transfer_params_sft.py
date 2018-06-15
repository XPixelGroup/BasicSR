import torch
from torch.nn import init

pretrained_net = torch.load('../../experiments/pretrained_models/SRGAN_bicx4_noBN_DIV2K.pth')
# should run train debug mode first to get an initial model
crt_net = torch.load('../../experiments/pretrained_models/sft_net_raw.pth')

for k, v in crt_net.items():
    if 'weight' in k:
        print(k, 'weight')
        init.kaiming_normal(v, a=0, mode='fan_in')
        v *= 0.1
    elif 'bias' in k:
        print(k, 'bias')
        v.fill_(0)

crt_net['conv0.weight'] = pretrained_net['model.0.weight']
crt_net['conv0.bias'] = pretrained_net['model.0.bias']
# residual blocks
for i in range(16):
    crt_net['sft_branch.{:d}.conv0.weight'.format(i)] = pretrained_net['model.1.sub.{:d}.res.0.weight'.format(i)]
    crt_net['sft_branch.{:d}.conv0.bias'.format(i)] = pretrained_net['model.1.sub.{:d}.res.0.bias'.format(i)]
    crt_net['sft_branch.{:d}.conv1.weight'.format(i)] = pretrained_net['model.1.sub.{:d}.res.2.weight'.format(i)]
    crt_net['sft_branch.{:d}.conv1.bias'.format(i)] = pretrained_net['model.1.sub.{:d}.res.2.bias'.format(i)]

crt_net['sft_branch.17.weight'] = pretrained_net['model.1.sub.16.weight']
crt_net['sft_branch.17.bias'] = pretrained_net['model.1.sub.16.bias']

# HR
crt_net['HR_branch.0.weight'] = pretrained_net['model.2.weight']
crt_net['HR_branch.0.bias'] = pretrained_net['model.2.bias']
crt_net['HR_branch.3.weight'] = pretrained_net['model.5.weight']
crt_net['HR_branch.3.bias'] = pretrained_net['model.5.bias']
crt_net['HR_branch.6.weight'] = pretrained_net['model.8.weight']
crt_net['HR_branch.6.bias'] = pretrained_net['model.8.bias']
crt_net['HR_branch.8.weight'] = pretrained_net['model.10.weight']
crt_net['HR_branch.8.bias'] = pretrained_net['model.10.bias']

print('OK. \n Saving model...')
torch.save(crt_net, '../../experiments/pretrained_models/sft_net_ini.pth')
