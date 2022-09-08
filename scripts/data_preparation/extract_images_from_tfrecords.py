import argparse
import cv2
import glob
import numpy as np
import os

from basicsr.utils.lmdb_util import LmdbMaker


def convert_celeba_tfrecords(tf_file, log_resolution, save_root, save_type='img', compress_level=1):
    """Convert CelebA tfrecords to images or lmdb files.

    Args:
        tf_file (str): Input tfrecords file in glob pattern.
            Example: 'datasets/celeba/celeba_tfrecords/validation/validation-r08-s-*-of-*.tfrecords'  # noqa:E501
        log_resolution (int): Log scale of resolution.
        save_root (str): Path root to save.
        save_type (str): Save type. Options: img | lmdb. Default: img.
        compress_level (int):  Compress level when encoding images. Default: 1.
    """
    if 'validation' in tf_file:
        phase = 'validation'
    else:
        phase = 'train'
    if save_type == 'lmdb':
        save_path = os.path.join(save_root, f'celeba_{2**log_resolution}_{phase}.lmdb')
        lmdb_maker = LmdbMaker(save_path)
    elif save_type == 'img':
        save_path = os.path.join(save_root, f'celeba_{2**log_resolution}_{phase}')
    else:
        raise ValueError('Wrong save type.')

    os.makedirs(save_path, exist_ok=True)

    idx = 0
    for record in sorted(glob.glob(tf_file)):
        print('Processing record: ', record)
        record_iterator = tf.python_io.tf_record_iterator(record)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # label = example.features.feature['label'].int64_list.value[0]
            # attr = example.features.feature['attr'].int64_list.value
            # male = attr[20]
            # young = attr[39]

            shape = example.features.feature['shape'].int64_list.value
            h, w, c = shape
            img_str = example.features.feature['data'].bytes_list.value[0]
            img = np.fromstring(img_str, dtype=np.uint8).reshape((h, w, c))

            img = img[:, :, [2, 1, 0]]

            if save_type == 'img':
                cv2.imwrite(os.path.join(save_path, f'{idx:08d}.png'), img)
            elif save_type == 'lmdb':
                _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
                key = f'{idx:08d}/r{log_resolution:02d}'
                lmdb_maker.put(img_byte, key, (h, w, c))

            idx += 1
            print(idx)

    if save_type == 'lmdb':
        lmdb_maker.close()


def convert_ffhq_tfrecords(tf_file, log_resolution, save_root, save_type='img', compress_level=1):
    """Convert FFHQ tfrecords to images or lmdb files.

    Args:
        tf_file (str): Input tfrecords file.
        log_resolution (int): Log scale of resolution.
        save_root (str): Path root to save.
        save_type (str): Save type. Options: img | lmdb. Default: img.
        compress_level (int):  Compress level when encoding images. Default: 1.
    """

    if save_type == 'lmdb':
        save_path = os.path.join(save_root, f'ffhq_{2**log_resolution}.lmdb')
        lmdb_maker = LmdbMaker(save_path)
    elif save_type == 'img':
        save_path = os.path.join(save_root, f'ffhq_{2**log_resolution}')
    else:
        raise ValueError('Wrong save type.')

    os.makedirs(save_path, exist_ok=True)

    idx = 0
    for record in sorted(glob.glob(tf_file)):
        print('Processing record: ', record)
        record_iterator = tf.python_io.tf_record_iterator(record)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            shape = example.features.feature['shape'].int64_list.value
            c, h, w = shape
            img_str = example.features.feature['data'].bytes_list.value[0]
            img = np.fromstring(img_str, dtype=np.uint8).reshape((c, h, w))

            img = img.transpose(1, 2, 0)
            img = img[:, :, [2, 1, 0]]
            if save_type == 'img':
                cv2.imwrite(os.path.join(save_path, f'{idx:08d}.png'), img)
            elif save_type == 'lmdb':
                _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
                key = f'{idx:08d}/r{log_resolution:02d}'
                lmdb_maker.put(img_byte, key, (h, w, c))

            idx += 1
            print(idx)

    if save_type == 'lmdb':
        lmdb_maker.close()


def make_ffhq_lmdb_from_imgs(folder_path, log_resolution, save_root, save_type='lmdb', compress_level=1):
    """Make FFHQ lmdb from images.

    Args:
        folder_path (str): Folder path.
        log_resolution (int): Log scale of resolution.
        save_root (str): Path root to save.
        save_type (str): Save type. Options: img | lmdb. Default: img.
        compress_level (int):  Compress level when encoding images. Default: 1.
    """

    if save_type == 'lmdb':
        save_path = os.path.join(save_root, f'ffhq_{2**log_resolution}_crop1.2.lmdb')
        lmdb_maker = LmdbMaker(save_path)
    else:
        raise ValueError('Wrong save type.')

    os.makedirs(save_path, exist_ok=True)

    img_list = sorted(glob.glob(os.path.join(folder_path, '*')))
    for idx, img_path in enumerate(img_list):
        print(f'Processing {idx}: ', img_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape

        if save_type == 'lmdb':
            _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
            key = f'{idx:08d}/r{log_resolution:02d}'
            lmdb_maker.put(img_byte, key, (h, w, c))

    if save_type == 'lmdb':
        lmdb_maker.close()


if __name__ == '__main__':
    """Read tfrecords w/o define a graph.

    We have tested it on TensorFlow 1.15

    References: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='ffhq', help="Dataset name. Options: 'ffhq' | 'celeba'. Default: 'ffhq'.")
    parser.add_argument(
        '--tf_file',
        type=str,
        default='datasets/ffhq/ffhq-r10.tfrecords',
        help=(
            'Input tfrecords file. For celeba, it should be glob pattern. '
            'Put quotes around the wildcard argument to prevent the shell '
            'from expanding it.'
            "Example: 'datasets/celeba/celeba_tfrecords/validation/validation-r08-s-*-of-*.tfrecords'"  # noqa:E501
        ))
    parser.add_argument('--log_resolution', type=int, default=10, help='Log scale of resolution.')
    parser.add_argument('--save_root', type=str, default='datasets/ffhq/', help='Save root path.')
    parser.add_argument(
        '--save_type', type=str, default='img', help="Save type. Options: 'img' | 'lmdb'. Default: 'img'.")
    parser.add_argument(
        '--compress_level', type=int, default=1, help='Compress level when encoding images. Default: 1.')
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except Exception:
        raise ImportError('You need to install tensorflow to read tfrecords.')

    if args.dataset == 'ffhq':
        convert_ffhq_tfrecords(
            args.tf_file,
            args.log_resolution,
            args.save_root,
            save_type=args.save_type,
            compress_level=args.compress_level)
    else:
        convert_celeba_tfrecords(
            args.tf_file,
            args.log_resolution,
            args.save_root,
            save_type=args.save_type,
            compress_level=args.compress_level)
