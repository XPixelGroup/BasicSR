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
mode = 1 # 1 for small data (more memory), 2 for large data (less memory)

img_list = sorted(glob.glob(img_folder))

print('Read images...')
# mode 1 small data, read all imgs
if mode == 1:
    dataset = [cv2.imread(v, cv2.IMREAD_UNCHANGED) for v in img_list]
    data_size = sum([img.nbytes for img in dataset])
# mode 2 large data, read imgs later
elif mode == 2:
    data_size = sum(os.stat(v).stat().st_size for v in img_list)
else:
    raise ValueError('mode should be 1 or 2')

env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

pbar = ProgressBar(len(img_list))
batch = 3000   # can be modified according to memory usage
txn = env.begin(write=True) # txn is a Transaction object
for i, v in enumerate(img_list):
    pbar.update('Write {}'.format(v))
    base_name = os.path.splitext(os.path.basename(v))[0]
    key = base_name.encode('ascii')
    data = dataset[i] if mode == 1 else cv2.imread(v, cv2.IMREAD_UNCHANGED)
    if data.ndim == 2:
        H, W = data.shape
        C = 1
    else:
        H, W, C = data.shape
    meta_key = (base_name + '.meta').encode('ascii')
    meta = '{:d}, {:d}, {:d}'.format(H, W, C)
    # The encode is only essential in Python 3
    txn.put(key, data)
    txn.put(meta_key, meta.encode('ascii'))
    if mode == 2 and i % batch == batch - 1:
        txn.commit()
        txn = env.begin(write=True)

txn.commit()
env.close()

print('Finish writing lmdb.')

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('Create lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys cache.')
