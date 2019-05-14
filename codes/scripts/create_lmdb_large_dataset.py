import sys
import os.path
import glob
import pickle
import lmdb
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar

# configurations
img_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800/*'  # glob matching pattern
lmdb_save_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800.lmdb'  # must end with .lmdb

img_list = sorted(glob.glob(img_folder))

print('Read images...', end=' ')
data_size = sum(os.stat(p).stat().st_size for p in img_list)
print('done!')

env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

pbar = ProgressBar(len(img_list))
batch = 3000   # can be modified according to memory usage
txn = env.begin(write=True)  # txn is a Transaction object
for i, v in enumerate(img_list):
    pbar.update('Write {}'.format(v))
    base_name = os.path.splitext(os.path.basename(v))[0]
    key = base_name.encode('ascii')
    data = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    if dataset[i].ndim == 2:
        H, W = dataset[i].shape
        C = 1
    else:
        H, W, C = data.shape
    meta_key = (base_name + '.meta').encode('ascii')
    meta = '{:d}, {:d}, {:d}'.format(H, W, C)
    # The encode is only essential in Python 3
    txn.put(key, data)
    txn.put(meta_key, meta.encode('ascii'))
    if i % batch == batch - 1:
        txn.commit()
        txn = env.begin(write=True)

print('Finish writing lmdb.')

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys cache.')
