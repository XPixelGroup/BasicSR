from basicsr.models.archs.hifacegan_arch import HiFaceGAN
from basicsr.models.archs.hifacegan_options import test_options
opt = test_options()
model = HiFaceGAN(opt)
print(model)
