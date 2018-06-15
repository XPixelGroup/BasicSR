import os
import os.path
import sys
import time
from multiprocessing import Pool
import numpy as np
import cv2


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def worker(GT_paths, save_GT_dir, mode):
    for GT_path in GT_paths:
        base_name = os.path.basename(GT_path)
        print(base_name, os.getpid())
        img_GT = cv2.imread(GT_path, cv2.IMREAD_UNCHANGED)  # BGR

        if mode == 'gray':
            img_y = cv2.cvtColor(img_GT, cv2.COLOR_BGR2GRAY)
        else:
            img_y = bgr2ycbcr(img_GT, only_y=True)

        cv2.imwrite(os.path.join(save_GT_dir, base_name), img_y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':

    GT_dir = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
    save_GT_dir = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_gray'
    mode = 'gray'  # 'y'
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
    def chunkify(lst, n):  # for non-continuous chunks
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
