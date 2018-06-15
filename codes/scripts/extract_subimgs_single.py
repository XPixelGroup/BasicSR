import os
import os.path
from multiprocessing import Pool
import time
import numpy as np
import cv2


def main():
    GT_dir = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800'
    save_GT_dir = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
    n_thread = 20

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
        p.apply_async(worker, args=(sub_lists[i], save_GT_dir))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    end = time.time()
    print('All subprocesses done. Using time {} sec.'.format(end - start))


def worker(GT_paths, save_GT_dir):
    crop_sz = 480
    step = 240
    thres_sz = 48

    for GT_path in GT_paths:
        base_name = os.path.basename(GT_path)
        print(base_name, os.getpid())
        img_GT = cv2.imread(GT_path, cv2.IMREAD_UNCHANGED)

        n_channels = len(img_GT.shape)
        if n_channels == 2:
            h, w = img_GT.shape
        elif n_channels == 3:
            h, w, c = img_GT.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)
        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                if n_channels == 2:
                    crop_img = img_GT[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img_GT[x:x + crop_sz, y:y + crop_sz, :]

                crop_img = np.ascontiguousarray(crop_img)
                index_str = '{:03d}'.format(index)
                # var = np.var(crop_img / 255)
                # if var > 0.008:
                #     print(index_str, var)
                cv2.imwrite(os.path.join(save_GT_dir, base_name.replace('.png', \
                    '_s'+index_str+'.png')), crop_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    main()
