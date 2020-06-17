import importlib

from basicsr.models.srgan_model import SRGANModel

loss_module = importlib.import_module('basicsr.models.losses')


class ESRGANModel(SRGANModel):
    """ESRGAN model for single image super-resolution.
    """
