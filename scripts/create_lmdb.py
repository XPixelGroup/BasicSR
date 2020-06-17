import os.path as osp
import sys
from multiprocessing import Pool

import cv2
import lmdb
import mmcv

from basicsr.utils import ProgressBar


def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb(folder_path, lmdb_path, img_path_list, keys)

    # LRx2 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X2_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb(folder_path, lmdb_path, img_path_list, keys)

    # LRx3 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X3_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 images
    folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def create_lmdb_for_reds():
    """Create lmdb files for REDS dataset.

    Usage:
        Before run this script, please run `merge_reds_train_val.py`.
        We take two folders for example:
            train_sharp
            train_sharp_bicubic
        Remember to modify opt configurations according to your settings.
    """
    # train_sharp
    folder_path = 'datasets/REDS/train_sharp'
    lmdb_path = 'datasets/REDS/train_sharp_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # train_sharp_bicubic
    folder_path = 'datasets/REDS/train_sharp_bicubic'
    lmdb_path = 'datasets/REDS/train_sharp_bicubic_with_val.lmdb'
    img_path_list, keys = prepare_keys_reds(folder_path)
    make_lmdb(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_reds(folder_path):
    """Prepare image path list and keys for REDS dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(mmcv.scandir(folder_path, suffix='png', recursive=True)))
    keys = [v.split('.png')[0] for v in img_path_list]  # example: 000/00000000

    return img_path_list, keys


def create_lmdb_for_vimeo90k():
    """Create lmdb files for Vimeo90K dataset.

    Usage:
        Remember to modify opt configurations according to your settings.
    """
    # GT
    folder_path = 'datasets/vimeo90k/vimeo_septuplet/sequences'
    lmdb_path = 'datasets/vimeo90k/vimeo90k_train_GT_only4th.lmdb'
    train_list_path = 'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path,
                                                'gt')
    make_lmdb(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)

    # LQ
    folder_path = 'datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
    lmdb_path = 'datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
    train_list_path = 'datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
    img_path_list, keys = prepare_keys_vimeo90k(folder_path, train_list_path,
                                                'lq')
    make_lmdb(
        folder_path, lmdb_path, img_path_list, keys, multiprocessing_read=True)


def prepare_keys_vimeo90k(folder_path, train_list_path, mode):
    """Prepare image path list and keys for Vimeo90K dataset.

    Args:
        folder_path (str): Folder path.
        train_list_path (str): Path to the official train list.
        mode (str): One of 'gt' or 'lq'.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    with open(train_list_path, 'r') as fin:
        train_list = [line.strip() for line in fin]

    img_path_list = []
    keys = []
    for line in train_list:
        folder, sub_folder = line.split('/')
        img_path_list.extend(
            [osp.join(folder, sub_folder, f'im{j + 1}.png') for j in range(7)])
        keys.extend([f'{folder}/{sub_folder}/im{j + 1}' for j in range(7)])

    if mode == 'gt':
        print('Only keep the 4th frame for the gt mode.')
        img_path_list = [v for v in img_path_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('/im4')]

    return img_path_list, keys


def make_lmdb(data_path,
              lmdb_path,
              img_path_list,
              keys,
              batch=5000,
              compress_level=1,
              multiprocessing_read=False,
              n_thread=40):
    """Make lmdb.

    Contents of lmdb. The file structure is:
    example.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records 1)image name (with extension),
    2)image shape, and 3)compression level, separated by a white space.

    For example, the meta information could be:
    `000_00000000.png (720,1280,3) 1`, which means:
    1) image name (with extension): 000_00000000.png;
    2) image shape: (720,1280,3);
    3) compression level: 1

    We use the image name without extension as the lmdb key.

    If `multiprocessing_read` is True, it will read all the images to memory
    using multiprocessing. Thus, your server needs to have enough memory.

    Args:
        data_path (str): Data path for reading images.
        lmdb_path (str): Lmdb save path.
        img_path_list (str): Image path list.
        keys (str): Used for lmdb keys.
        batch (int): After processing batch images, lmdb commits.
            Default: 5000.
        compress_level (int): Compress level when encoding images. Default: 1.
        multiprocessing_read (bool): Whether use multiprocessing to read all
            the images to memory. Default: False.
        n_thread (int): For multiprocessing.
    """
    assert len(img_path_list) == len(keys), (
        'img_path_list and keys should have the same length, '
        f'but got {len(img_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Totoal images: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    if multiprocessing_read:
        # read all the images to memory (multiprocessing)
        dataset = {}  # use dict to keep the order for multiprocessing
        shapes = {}
        print(f'Read images with multiprocessing, #thread: {n_thread} ...')
        pbar = ProgressBar(len(img_path_list))

        def callback(arg):
            """get the image data and update pbar."""
            key, dataset[key], shapes[key] = arg
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(
                read_img_worker,
                args=(osp.join(data_path, path), key, compress_level),
                callback=callback)
        pool.close()
        pool.join()
        print(f'Finish reading {len(img_path_list)} images.')

    # create lmdb environment
    # obtain data size for one image
    img = mmcv.imread(osp.join(data_path, img_path_list[0]), flag='unchanged')
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    data_size_per_img = img_byte.nbytes
    print('Data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(img_path_list)
    env = lmdb.open(lmdb_path, map_size=data_size * 10)

    # write data to lmdb
    pbar = ProgressBar(len(img_path_list))
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(f'Write {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(
                osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # write meta information
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')


def read_img_worker(path, key, compress_level):
    """Read image worker

    Args:
        path (str): Image path.
        key (str): Image key.
        compress_level (int): Compress level when encoding images.

    Returns:
        str: Image key.
        byte: Image byte.
        tuple[int]: Image shape.
    """
    img = mmcv.imread(path, flag='unchanged')
    if img.ndim == 2:
        h, w = img.shape
        c = 1
    else:
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img,
                               [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


if __name__ == '__main__':
    create_lmdb_for_div2k()
    # create_lmdb_for_reds()
    # create_lmdb_for_vimeo90k()
