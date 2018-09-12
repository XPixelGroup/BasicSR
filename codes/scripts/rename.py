import os
import os.path
import glob


input_folder = '/home/xtwang/Projects/PIRM18/results/pirm_selfval_img06/*'  # glob matching pattern
save_folder = '/home/xtwang/Projects/PIRM18/results/pirm_selfval_img'

mode = 'cp'  # 'cp' | 'mv'
file_list = sorted(glob.glob(input_folder))

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print('mkdir ... ' + save_folder)
else:
    print('File [{}] already exists. Exit.'.format(save_folder))

for i, path in enumerate(file_list):
    base_name = os.path.splitext(os.path.basename(path))[0]

    new_name = base_name.split('_')[0]
    new_path = os.path.join(save_folder, new_name + '.png')

    os.system(mode + ' ' + path + ' ' + new_path)
    print(i, base_name)
