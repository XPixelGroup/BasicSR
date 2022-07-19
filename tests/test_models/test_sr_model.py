import tempfile
import torch
import yaml

from basicsr.archs.srresnet_arch import MSRResNet
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.losses.basic_loss import L1Loss, PerceptualLoss
from basicsr.models.sr_model import SRModel


def test_srmodel():
    """Test model: SRModel"""

    opt_str = r"""
scale: 4
num_gpu: 1
manual_seed: 0
is_train: True
dist: False

# network structures
network_g:
  type: MSRResNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 4
  num_block: 1
  upscale: 4

# path
path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

# training settings
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: CosineAnnealingRestartLR
    periods: [250000, 250000, 250000, 250000]
    restart_weights: [1, 1, 1, 1]
    eta_min: !!float 1e-7

  total_iter: 1000000
  warmup_iter: -1  # no warm up

  # losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean
  perceptual_opt:
    type: PerceptualLoss
    layer_weights:
      'conv5_4': 1  # before relu
    vgg_type: vgg19
    use_input_norm: true
    range_norm: false
    perceptual_weight: 1.0
    style_weight: 1.0
    criterion: l1

# validation settings
val:
  val_freq: !!float 5e3
  save_img: True

  metrics:
    psnr: # metric name
      type: calculate_psnr
      crop_border: 4
      test_y_channel: false
      better: higher  # the higher, the better. Default: higher
"""

    opt = yaml.safe_load(opt_str)

    # build model
    model = SRModel(opt)
    # test attributes
    assert model.__class__.__name__ == 'SRModel'
    assert isinstance(model.net_g, MSRResNet)
    assert isinstance(model.cri_pix, L1Loss)
    assert isinstance(model.cri_perceptual, PerceptualLoss)
    assert isinstance(model.optimizers[0], torch.optim.Adam)
    assert model.ema_decay == 0.999

    # prepare data
    gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    lq = torch.rand((1, 3, 8, 8), dtype=torch.float32)
    data = dict(gt=gt, lq=lq)
    model.feed_data(data)
    # check data shape
    assert model.lq.shape == (1, 3, 8, 8)
    assert model.gt.shape == (1, 3, 32, 32)

    # ----------------- test optimize_parameters -------------------- #
    model.optimize_parameters(1)
    assert model.output.shape == (1, 3, 32, 32)
    assert isinstance(model.log_dict, dict)
    # check returned keys
    expected_keys = ['l_pix', 'l_percep', 'l_style']
    assert set(expected_keys).issubset(set(model.log_dict.keys()))

    # ----------------- test save -------------------- #
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['path']['models'] = tmpdir
        model.opt['path']['training_states'] = tmpdir
        model.save(0, 1)

    # ----------------- test the test function -------------------- #
    model.test()
    assert model.output.shape == (1, 3, 32, 32)
    # delete net_g_ema
    model.__delattr__('net_g_ema')
    model.test()
    assert model.output.shape == (1, 3, 32, 32)
    assert model.net_g.training is True  # should back to training mode after testing

    # ----------------- test nondist_validation -------------------- #
    # construct dataloader
    dataset_opt = dict(
        name='Test',
        dataroot_gt='tests/data/gt',
        dataroot_lq='tests/data/lq',
        io_backend=dict(type='disk'),
        scale=4,
        phase='val')
    dataset = PairedImageDataset(dataset_opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    assert model.is_train is True
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['path']['visualization'] = tmpdir
        model.nondist_validation(dataloader, 1, None, save_img=True)
        assert model.is_train is True
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)

    # in validation mode
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['is_train'] = False
        model.opt['val']['suffix'] = 'test'
        model.opt['path']['visualization'] = tmpdir
        model.opt['val']['pbar'] = True
        model.nondist_validation(dataloader, 1, None, save_img=True)
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)

        # if opt['val']['suffix'] is None
        model.opt['val']['suffix'] = None
        model.opt['name'] = 'demo'
        model.opt['path']['visualization'] = tmpdir
        model.nondist_validation(dataloader, 1, None, save_img=True)
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)
