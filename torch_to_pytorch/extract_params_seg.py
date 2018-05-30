import torch
from torch.utils.serialization import load_lua


net = load_lua('./torch_models/OutdoorSceneSeg_bic_iter_30000.t7')
print(net)
py_model = torch.load('./pytorch_models/OutdoorSceneSeg_bic_iter_30000.pth')

for k, v in py_model.items():
    print(k)
ori_keys = len(py_model)

def load_ConvBN(conv_name, bn_name, t7):
    # conv
    py_model[conv_name + '.weight'] = t7.modules[0].weight
    # BN
    py_model[bn_name + '.weight'] = t7.modules[1].weight
    py_model[bn_name + '.bias'] = t7.modules[1].bias
    py_model[bn_name + '.running_mean'] = t7.modules[1].running_mean
    py_model[bn_name + '.running_var'] = t7.modules[1].running_var

def load_res131(block_name, t7, has_proj=False):
    # res
    load_ConvBN(block_name + '.res.0', block_name + '.res.1', t7.modules[0].modules[0].modules[0])
    load_ConvBN(block_name + '.res.3', block_name + '.res.4', t7.modules[0].modules[0].modules[1])
    load_ConvBN(block_name + '.res.6', block_name + '.res.7', t7.modules[0].modules[0].modules[2])
    if has_proj:
        load_ConvBN(block_name + '.proj.0', block_name + '.proj.1', t7.modules[0].modules[1].modules[0])

# conv1
load_ConvBN('feature.0', 'feature.1', net.modules[0])
load_ConvBN('feature.3', 'feature.4', net.modules[1])
load_ConvBN('feature.6', 'feature.7', net.modules[2])
# conv2
load_res131('feature.10', net.modules[4], True)
load_res131('feature.11', net.modules[5], False)
load_res131('feature.12', net.modules[6], False)
# conv3
load_res131('feature.13', net.modules[7], True)
load_res131('feature.14', net.modules[8], False)
load_res131('feature.15', net.modules[9], False)
load_res131('feature.16', net.modules[10], False)
# conv4
load_res131('feature.17', net.modules[11], True)
for i in range(22):
    load_res131('feature.{:d}'.format(18 + i), net.modules[12 + i], False)
# conv5
load_res131('feature.40', net.modules[34], True)
load_res131('feature.41', net.modules[35], False)
load_res131('feature.42', net.modules[36], False)
load_ConvBN('feature.43', 'feature.44', net.modules[37])
# conv6
py_model['feature.47.weight'] = net.modules[39].weight
py_model['feature.47.bias'] = net.modules[39].bias
# deconv
deconv_weight = py_model['deconv.weight']
for i in range(8):
    deconv_weight[i, :, :, :] = net.modules[40].weight[i, i, :, :].unsqueeze(0)

assert ori_keys == len(py_model), 'the keys mismatch! Please check.'

torch.save(py_model, '../experiments/pretrained_models/OutdoorSceneSeg_bic.pth')
