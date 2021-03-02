import torch
import torch.nn as nn
import lpips  # pip install lpips


class LPIPS(nn.Module):

    def __init__(self, net='alex', verbose=True):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net, verbose=verbose)
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x, y):
        lpips_value = self.lpips(x, y, normalize=True)
        return lpips_value.mean()


def calculate_lpips(x, y, function=None, **kwargs):
    return function(x, y).mean()
