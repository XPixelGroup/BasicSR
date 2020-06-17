import importlib
import os.path as osp

import mmcv
import torch
import torch.utils.data

from basicsr.utils import get_root_logger

__all__ = ['create_dataset', 'create_dataloader']

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in mmcv.scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'basicsr.data.{file_name}')
    for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """Create dataset.

    Args:
        dataset_opt (dict): Configuration for dataset. It constains:
            name (str): Dataset name.
            type (str): Dataset type.
    """
    dataset_type = dataset_opt['type']

    # dynamically instantiation
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    dataset = dataset_cls(dataset_opt)

    logger = get_root_logger()
    logger.info(
        f'Dataset {dataset.__class__.__name__} - {dataset_opt["name"]} '
        'is created.')
    return dataset


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker (int): Number of workers for each GPU.
            batch_size (int): Training batch size for all GPUs.
        opt (dict): Config options. Default: None.
        It contains the following keys:
            dist (bool): Distributed training or not.
            gpu_ids (list): GPU indexes.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
    """
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:  # distributed training
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['num_worker']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:  # non-distributed training
            num_workers = dataset_opt['num_worker'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=False)
    else:  # validation
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)
