import pytest
import torch

from basicsr.archs.duf_arch import DUF, DynamicUpsamplingFilter


def test_duf():
    """Test arch: DUF."""

    # model init and forward
    net = DUF(scale=4, num_layer=16, adapt_official_weights=False).cuda()
    img = torch.rand((1, 7, 3, 48, 48), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 3, 192, 192)

    # ----------------- test scale x3, num_layer=28 ---------------------- #
    net = DUF(scale=3, num_layer=28, adapt_official_weights=True).cuda()
    output = net(img)
    assert output.shape == (1, 3, 144, 144)

    # ----------------- test scale x2, num_layer=52 ---------------------- #
    net = DUF(scale=2, num_layer=52, adapt_official_weights=True).cuda()
    output = net(img)
    assert output.shape == (1, 3, 96, 96)

    # ----------------- unsupported num_layers ---------------------- #
    with pytest.raises(ValueError):
        net = DUF(scale=2, num_layer=4, adapt_official_weights=True)


def test_dynamicupsamplingfilter():
    """Test block: DynamicUpsamplingFilter"""
    net = DynamicUpsamplingFilter(filter_size=(3, 3)).cuda()
    img = torch.rand((2, 3, 12, 12), dtype=torch.float32).cuda()
    filters = torch.rand((2, 9, 2, 12, 12), dtype=torch.float32).cuda()
    output = net(img, filters)
    assert output.shape == (2, 6, 12, 12)

    # ----------------- wrong filter_size type ---------------------- #
    with pytest.raises(TypeError):
        DynamicUpsamplingFilter(filter_size=4)

    # ----------------- wrong filter_size shape ---------------------- #
    with pytest.raises(ValueError):
        DynamicUpsamplingFilter(filter_size=(3, 3, 3))
