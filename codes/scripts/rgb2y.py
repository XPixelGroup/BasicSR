import os
import os.path
import sys
import time
from multiprocessing import Pool
import numpy as np
import cv2


def rgb2ycbcr(img, only_y=True):
    # the same as matlab rgb2ycbcr
    # TODO support double [0, 1]
    assert img.dtype == np.uint8, 'np.uint8 is supposed. But received img dtype: %s.' % img.dtype
    in_img_type = img.dtype
    img.astype(np.float64)
    if only_y:  # only return Y channel
        img_y = (np.dot(img[..., :3], [65.481, 128.553, 24.966]) / 255.0 + 16.0).round()
        return img_y.astype(in_img_type)
    else:
        img_ycbcr = (np.matmul(img[..., :3], [[65.481, -37.797, 112.0], \
        [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]).round()
        return img_ycbcr.astype(in_img_type)


def rgb2gray(img):
    '''
    rgb2gray is the Y component of YUV;
    the same as matlab rgb2gray where coefficients are [0.2989, 0.587, 0.114]
    Input: image Numpy array, [0,255], HWC, RGB
    Output: image Numpy array, [0, 255], HW
    '''
    assert img.dtype == np.uint8, 'np.uint8 is supposed. But received img dtype: %s.' % img.dtype
    in_img_type = img.dtype
    img.astype(np.float64)
    img_gray = np.dot(img[..., :3], [0.2989, 0.587, 0.114]).round()
    return img_gray.astype(in_img_type)


def worker(GT_paths, save_GT_dir, mode):
    for GT_path in GT_paths:
        base_name = os.path.basename(GT_path)
        print(base_name, os.getpid())
        img_GT = cv2.imread(GT_path, cv2.IMREAD_UNCHANGED)
        img_GT = img_GT[:, :, [2, 1, 0]]

        if mode == 'gray':
            func = rgb2gray
        else:
            func = rgb2ycbcr
        img_y = func(img_GT)
        cv2.imwrite(os.path.join(save_GT_dir, base_name), img_y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__=='__main__':

    GT_dir = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
    save_GT_dir = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_y'
    mode = 'gray' # 'y'
    n_thread = 20

    if not os.path.exists(save_GT_dir):
        os.makedirs(save_GT_dir)
        print('mkdir ... ' + save_GT_dir)
    else:
        print('File [%s] already exists. Exit.' % save_GT_dir)
        sys.exit(1)

    print('Parent process %s.' % os.getpid())
    start = time.time()

    p = Pool(n_thread)
    # read all files to a list
    all_files = []
    for root, _, fnames in sorted(os.walk(GT_dir)):
        full_path = [os.path.join(root, x) for x in fnames]
        all_files.extend(full_path)
    # cut into subtasks
    def chunkify(lst, n): # for non-continuous chunks
        return [lst[i::n] for i in range(n)]
    sub_lists = chunkify(all_files, n_thread)
    # call workers
    for i in range(n_thread):
        p.apply_async(worker, args=(sub_lists[i], save_GT_dir, mode))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    end = time.time()
    print('All subprocesses done. Using time {} sec.'.format(end - start))
