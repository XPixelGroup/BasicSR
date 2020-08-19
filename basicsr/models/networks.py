from basicsr.models.archs import _arch_modules


def dynamic_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
            files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class： Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)


# Generator
def define_net_g(opt):
    network_type = opt.pop('type')
    net_g = dynamic_instantiation(_arch_modules, network_type, opt)

    return net_g


# Discriminator
def define_net_d(opt):
    network_type = opt.pop('type')

    net_d = dynamic_instantiation(_arch_modules, network_type, opt)
    return net_d
