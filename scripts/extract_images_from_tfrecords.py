"""Read tfrecords w/o define a graph.

Ref:
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
"""

import cv2
import glob
import numpy as np
import os

from basicsr.utils.lmdb import LmdbMaker


def celeba_tfrecords():
    # Configurations
    file_pattern = '/home/xtwang/datasets/CelebA_tfrecords/celeba-full-tfr/train/train-r08-s-*-of-*.tfrecords'  # noqa:E501
    # r08: resolution 2^8 = 256
    resolution = 128
    save_path = f'/home/xtwang/datasets/CelebA_tfrecords/tmptrain_{resolution}'

    save_all_path = os.path.join(save_path, f'all_{resolution}')
    os.makedirs(save_all_path)

    idx = 0
    print(glob.glob(file_pattern))
    for record in glob.glob(file_pattern):
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

            # save image
            img = img[:, :, [2, 1, 0]]
            cv2.imwrite(os.path.join(save_all_path, f'{idx:08d}.png'), img)

            idx += 1
            print(idx)


def ffhq_tfrecords():
    # Configurations
    file_pattern = '/home/xtwang/datasets/ffhq/ffhq-r10.tfrecords'
    resolution = 1024
    save_path = f'/home/xtwang/datasets/ffhq/ffhq_imgs/ffhq_{resolution}'

    os.makedirs(save_path, exist_ok=True)
    idx = 0
    print(glob.glob(file_pattern))
    for record in glob.glob(file_pattern):
        record_iterator = tf.python_io.tf_record_iterator(record)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            shape = example.features.feature['shape'].int64_list.value
            c, h, w = shape
            img_str = example.features.feature['data'].bytes_list.value[0]
            img = np.fromstring(img_str, dtype=np.uint8).reshape((c, h, w))

            # save image
            img = img.transpose(1, 2, 0)
            img = img[:, :, [2, 1, 0]]
            cv2.imwrite(os.path.join(save_path, f'{idx:08d}.png'), img)

            idx += 1
            print(idx)


def ffhq_tfrecords_to_lmdb():
    # Configurations
    file_pattern = '/home/xtwang/datasets/ffhq/ffhq-r10.tfrecords'
    log_resolution = 10
    compress_level = 1
    lmdb_path = f'/home/xtwang/datasets/ffhq/ffhq_{2**log_resolution}.lmdb'

    idx = 0
    print(glob.glob(file_pattern))

    lmdb_maker = LmdbMaker(lmdb_path)
    for record in glob.glob(file_pattern):
        record_iterator = tf.python_io.tf_record_iterator(record)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            shape = example.features.feature['shape'].int64_list.value
            c, h, w = shape
            img_str = example.features.feature['data'].bytes_list.value[0]
            img = np.fromstring(img_str, dtype=np.uint8).reshape((c, h, w))

            # write image to lmdb
            img = img.transpose(1, 2, 0)
            img = img[:, :, [2, 1, 0]]
            _, img_byte = cv2.imencode(
                '.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
            key = f'{idx:08d}/r{log_resolution:02d}'
            lmdb_maker.put(img_byte, key, (h, w, c))

            idx += 1
            print(key)
    lmdb_maker.close()


if __name__ == '__main__':
    # we have test on TensorFlow 1.15
    try:
        import tensorflow as tf
    except Exception:
        raise ImportError('You need to install tensorflow to read tfrecords.')
    # celeba_tfrecords()
    # ffhq_tfrecords()
    ffhq_tfrecords_to_lmdb()
