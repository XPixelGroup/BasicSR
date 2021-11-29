import yaml

from basicsr.data.paired_image_dataset import PairedImageDataset


def test_pairedimagedataset():
    """Test dataset: PairedImageDataset"""

    opt_str = r"""
name: Test
type: PairedImageDataset
dataroot_gt: tests/data/gt
dataroot_lq: tests/data/lq
meta_info_file: tests/data/meta_info_gt.txt
filename_tmpl: '{}'
io_backend:
    type: disk

scale: 4
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
gt_size: 128
use_hflip: true
use_rot: true

phase: train
"""
    opt = yaml.safe_load(opt_str)

    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info
    assert dataset.mean == [0.5, 0.5, 0.5]

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 128, 128)
    assert result['lq'].shape == (3, 32, 32)
    assert result['lq_path'] == 'tests/data/lq/baboon.png'
    assert result['gt_path'] == 'tests/data/gt/baboon.png'

    # ------------------ test filename_tmpl -------------------- #
    opt.pop('filename_tmpl')
    opt['io_backend'] = dict(type='disk')
    dataset = PairedImageDataset(opt)
    assert dataset.filename_tmpl == '{}'

    # ------------------ test scan folder mode -------------------- #
    opt.pop('meta_info_file')
    opt['io_backend'] = dict(type='disk')
    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to correctly scan folders

    # ------------------ test lmdb backend and with y channel-------------------- #
    opt['dataroot_gt'] = 'tests/data/gt.lmdb'
    opt['dataroot_lq'] = 'tests/data/lq.lmdb'
    opt['io_backend'] = dict(type='lmdb')
    opt['color'] = 'y'
    opt['mean'] = [0.5]
    opt['std'] = [0.5]

    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info
    assert dataset.std == [0.5]

    # test __getitem__
    result = dataset.__getitem__(1)
    # check returned keys
    expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (1, 128, 128)
    assert result['lq'].shape == (1, 32, 32)
    assert result['lq_path'] == 'comic'
    assert result['gt_path'] == 'comic'

    # ------------------ test case: val/test mode -------------------- #
    opt['phase'] = 'test'
    opt['io_backend'] = dict(type='lmdb')
    dataset = PairedImageDataset(opt)

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (1, 480, 492)
    assert result['lq'].shape == (1, 120, 123)
    assert result['lq_path'] == 'baboon'
    assert result['gt_path'] == 'baboon'
