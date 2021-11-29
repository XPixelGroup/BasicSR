import yaml

from basicsr.data.single_image_dataset import SingleImageDataset


def test_singleimagedataset():
    """Test dataset: SingleImageDataset"""

    opt_str = r"""
name: Test
type: SingleImageDataset
dataroot_lq: tests/data/lq
meta_info_file: tests/data/meta_info_gt.txt
io_backend:
    type: disk

mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
"""
    opt = yaml.safe_load(opt_str)

    dataset = SingleImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info
    assert dataset.mean == [0.5, 0.5, 0.5]

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['lq', 'lq_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['lq'].shape == (3, 120, 123)
    assert result['lq_path'] == 'tests/data/lq/baboon.png'

    # ------------------ test scan folder mode -------------------- #
    opt.pop('meta_info_file')
    opt['io_backend'] = dict(type='disk')
    dataset = SingleImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to correctly scan folders

    # ------------------ test lmdb backend and with y channel-------------------- #
    opt['dataroot_lq'] = 'tests/data/lq.lmdb'
    opt['io_backend'] = dict(type='lmdb')
    opt['color'] = 'y'
    opt['mean'] = [0.5]
    opt['std'] = [0.5]

    dataset = SingleImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info
    assert dataset.std == [0.5]

    # test __getitem__
    result = dataset.__getitem__(1)
    # check returned keys
    expected_keys = ['lq', 'lq_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['lq'].shape == (1, 90, 60)
    assert result['lq_path'] == 'comic'
