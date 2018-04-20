import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable


# Assume input in range [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=11, use_bn=True, use_input_norm=True):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        print(model)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1), requires_grad=False)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1), requires_grad=False)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to bp to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False
        print(self.features)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output