import torch
try:
    import torch_xla.core.xla_model as xm
except:
    pass

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
    accelerator = accelerator_name(opt)
    if accelerator == 'xla':
        # Devices of the same hw family.
        # Note: returns 1 when replication is in place!
        # device = xm.xla_device()
        # devices = xm.get_xla_supported_devices(xm.xla_device_hw(device))
        # return len(devices)

        # This works when replication is active
        return xm.xrt_world_size()
    return torch.cuda.device_count()

def use_xmp(opt):
    accelerator = accelerator_name(opt)
    if accelerator != 'xla':
        return False
    return device_count(opt) > 1