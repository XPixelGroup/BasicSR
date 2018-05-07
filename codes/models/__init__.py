def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'noisesr':
        from .NoiseSR_model import NoiseSRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_model import SFTModel as M
    else:
        raise NotImplementedError('Model [%s] not recognized.' % model)
    m = M(opt)
    print('Model [%s] is created.' % m.name())
    return m