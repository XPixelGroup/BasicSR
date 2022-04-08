import pytest
import torch

from basicsr.archs.ecbsr_arch import ECB, ECBSR, SeqConv3x3


def test_ecbsr():
    """Test arch: ECBSR."""

    # model init and forward
    net = ECBSR(num_in_ch=1, num_out_ch=1, num_block=1, num_channel=4, with_idt=False, act_type='prelu', scale=4).cuda()
    img = torch.rand((1, 1, 12, 12), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 1, 48, 48)

    # ----------------- test 3 channels ---------------------- #
    net = ECBSR(num_in_ch=3, num_out_ch=3, num_block=1, num_channel=4, with_idt=True, act_type='rrelu', scale=2).cuda()
    img = torch.rand((1, 3, 12, 12), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 3, 24, 24)


def test_seqconv3x3():
    """Test block: SeqConv3x3."""

    # model init and forward
    net = SeqConv3x3(seq_type='conv1x1-conv3x3', in_channels=2, out_channels=2, depth_multiplier=2).cuda()
    img = torch.rand((1, 2, 12, 12), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 2, 12, 12)
    # test rep_params
    conv = torch.nn.Conv2d(2, 2, 3, 1, 1).cuda()
    conv.weight.data, conv.bias.data = net.rep_params()
    output_rep = conv(img)
    assert output_rep.shape == (1, 2, 12, 12)
    # whether the two results are close
    assert torch.allclose(output, output_rep, rtol=1e-5, atol=1e-5)

    # ----------------- test rep_params with conv1x1-laplacian seq ---------------------- #
    net = SeqConv3x3(seq_type='conv1x1-laplacian', in_channels=4, out_channels=4, depth_multiplier=3).cuda()
    img = torch.rand((1, 4, 12, 12), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 4, 12, 12)
    # test rep_params
    conv = torch.nn.Conv2d(4, 4, 3, 1, 1).cuda()
    conv.weight.data, conv.bias.data = net.rep_params()
    output_rep = conv(img)
    assert output_rep.shape == (1, 4, 12, 12)
    # whether the two results are close
    assert torch.allclose(output, output_rep, rtol=1e-5, atol=1e-5)

    # ----------------- unsupported type ---------------------- #
    with pytest.raises(ValueError):
        SeqConv3x3(seq_type='noseq', in_channels=1, out_channels=1)


def test_ecb():
    """Test block: ECB."""
    # model init and forward
    net = ECB(in_channels=2, out_channels=2, depth_multiplier=1, act_type='softplus', with_idt=False).cuda()
    img = torch.rand((1, 2, 12, 12), dtype=torch.float32).cuda()
    output = net(img)
    assert output.shape == (1, 2, 12, 12)
    # test rep_params
    net = net.eval()
    output_rep = net(img)
    assert output_rep.shape == (1, 2, 12, 12)
    # whether the two results are close
    assert torch.allclose(output, output_rep, rtol=1e-5, atol=1e-5)

    # ----------------- linear activation function and identity---------------------- #
    net = ECB(in_channels=2, out_channels=2, depth_multiplier=1, act_type='linear', with_idt=True).cuda()
    output = net(img)
    assert output.shape == (1, 2, 12, 12)
    # test rep_params
    net = net.eval()
    output_rep = net(img)
    assert output_rep.shape == (1, 2, 12, 12)
    # whether the two results are close
    assert torch.allclose(output, output_rep, rtol=1e-5, atol=1e-5)

    # ----------------- relu activation function---------------------- #
    net = ECB(in_channels=2, out_channels=2, depth_multiplier=1, act_type='relu', with_idt=False).cuda()
    output = net(img)
    assert output.shape == (1, 2, 12, 12)

    # ----------------- unsupported type ---------------------- #
    with pytest.raises(ValueError):
        ECB(in_channels=2, out_channels=2, depth_multiplier=1, act_type='unknown', with_idt=False)
