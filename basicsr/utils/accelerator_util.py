import torch
try:
    import torch_xla.core.xla_model as xm
except:
    pass

        # self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')

def accelerator_name(opt):
    if opt['num_gpu'] == 0:
        return 'cpu'
    return opt.get('accelerator', 'cuda')

def default_device(opt):
    accelerator = accelerator_name(opt)
    if accelerator == 'xla':
        return xm.xla_device()
    return accelerator

def device_count(opt):
    accelerator = opt.get('accelerator', 'cuda')
    if accelerator == 'xla':
        device = xm.xla_device()
        return 0 if device is None else 1
    return torch.cuda.device_count()