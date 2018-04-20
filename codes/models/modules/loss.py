import torch
import torch.nn as nn
from torch.autograd import Variable


# Define GAN loss: [vanilla | lsgan | wgan]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type in ['vanilla', 'lsgan']:
            self.register_buffer('real_label', torch.Tensor())
            self.register_buffer('fake_label', torch.Tensor())

        # print(type(real_label), type(fake_label))

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [%s] is not found' % self.gan_type)

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan':
            return target_is_real
        if target_is_real:
            if self.real_label.size() != input.size():  # check if new label needed
                self.real_label.resize_(input.size()).fill_(self.real_label_val)
            return Variable(self.real_label)
        else:
            if self.fake_label.size() != input.size():  # check if new label needed
                self.fake_label.resize_(input.size()).fill_(self.fake_label_val)
            return Variable(self.fake_label)

    def forward(self, input, target_is_real):
        # start_time = time.time()
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        # print('GANLoss time:', time.time() - start_time)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        # print(grad_interp_norm)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss