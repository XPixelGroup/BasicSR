import torch


class TorchDeviceFactory:
    @classmethod
    def get(cls, opt):
        # initialize model
        if opt['num_gpu'] != 0 and torch.cuda.is_available():
            print(f'Using cuda.')
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            print('Using mps.')
            return torch.device('mps')
        else:
            print('Using cpu.')
            return torch.device('cpu')
