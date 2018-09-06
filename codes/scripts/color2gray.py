import os
import os.path
import sys
from multiprocessing import Pool
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.util import bgr2ycbcr
from utils.progress_bar import ProgressBar


def main():
    """A multi-thread tool for converting RGB images to gary/Y images."""

    input_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800'
    save_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_gray'
    mode = 'gray'  # 'gray' | 'y': Y channel in YCbCr space
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    n_thread = 20  # thread number

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    # print('Parent process {:d}.'.format(os.getpid()))

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker, args=(path, save_folder, mode, compression_level), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder, mode, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR
    if mode == 'gray':
        img_y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_y = bgr2ycbcr(img, only_y=True)
    cv2.imwrite(
        os.path.join(save_folder, img_name), img_y,
        [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
