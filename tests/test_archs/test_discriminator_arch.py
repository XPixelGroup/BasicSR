import pytest
import torch

from basicsr.archs.discriminator_arch import VGGStyleDiscriminator


def test_vggstylediscriminator():
    """Test arch: VGGStyleDiscriminator."""

    # model init and forward
    net = VGGStyleDiscriminator(num_in_ch=3, num_feat=4).cuda()
    img = torch.rand((1, 3, 128, 128), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 1)

    # ----------------------- input_size is 256 x 256------------------------ #
    net = VGGStyleDiscriminator(num_in_ch=3, num_feat=4, input_size=256).cuda()
    img = torch.rand((1, 3, 256, 256), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 1)

    # ----------------------- input feature size is not identical to input_size------------------------- #
    with pytest.raises(AssertionError):
        img = torch.rand((1, 3, 128, 128), dtype=torch.float32).cuda()
        output = net(img)

    # ----------------------- input_size is not 128 or 256------------------------- #
    with pytest.raises(AssertionError):
        net = VGGStyleDiscriminator(num_in_ch=3, num_feat=4, input_size=64)
