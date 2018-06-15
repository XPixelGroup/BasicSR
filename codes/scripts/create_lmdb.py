import os.path
import glob
import pickle
import lmdb
import cv2

# configurations
img_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub/*'  # glob matching pattern
lmdb_save_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb'  # must end with .lmdb

file_list = sorted(glob.glob(img_path))
dataset = []
data_size = 0

print('Read image...')
for i, v in enumerate(file_list):
    if i % 100 == 0:
        print(i)
    img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
    dataset.append(img)
    data_size += img.nbytes
map_size = data_size * 10
env = lmdb.open(lmdb_save_path, map_size=map_size)

print('Finish reading image - {}.\nWrite to lmdb...'.format(len(file_list)))
with env.begin(write=True) as txn:  # txn is a Transaction object
    for i, v in enumerate(file_list):
        if i % 100 == 0:
            print(i)
        img_name = os.path.splitext(os.path.basename(v))[0]
        key = img_name.encode('ascii')
        data = dataset[i]
        if dataset[i].ndim == 2:
            H, W = dataset[i].shape
            C = 1
        else:
            H, W, C = dataset[i].shape
        meta_key = (img_name + '.meta').encode('ascii')
        meta = '{:d}, {:d}, {:d}'.format(H, W, C)
        # The encode is only essential in Python 3
        txn.put(key, data)
        txn.put(meta_key, meta.encode('ascii'))
print('Finish writing to lmdb - {}.'.format(len(file_list)))

# create keys cache
keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print('creating lmdb keys cache: {}'.format(keys_cache_file))
    keys = [key.decode('ascii') for key, _ in txn.cursor()]
    pickle.dump(keys, open(keys_cache_file, "wb"))
print('Finish creating lmdb keys.')
