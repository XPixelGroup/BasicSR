def create_model(opt):
    model = opt['model']
    if model == 'test_sr_srgan':
        from .test_G_model import TestGModel as M
    elif model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'train_srgan':
        from .train_SRGAN_model import TrainSRGANModel as M
    else:
        raise NotImplementedError('Model [%s] not recognized.' % model)
    m = M(opt)
    print('Model [%s] is created.' % m.name())
    return m