import torch

from basicsr.archs.srresnet_arch import MSRResNet


def test_msrresnet():
    """Test arch: MSRResNet."""

    # model init and forward
    net = MSRResNet(num_in_ch=3, num_out_ch=3, num_feat=12, num_block=2, upscale=4).cuda()
    img = torch.rand((1, 3, 16, 16), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 3, 64, 64)

    # ----------------- the x3 case ---------------------- #
    net = MSRResNet(num_in_ch=1, num_out_ch=1, num_feat=4, num_block=1, upscale=3).cuda()
    img = torch.rand((1, 1, 16, 16), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 1, 48, 48)
