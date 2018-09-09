import os.path
import glob
import cv2

crt_path = os.path.dirname(os.path.realpath(__file__))

# configurations
h_start, h_len = 170, 64
w_start, w_len = 232, 100
enlarge_ratio = 3
line_width = 2
color = 'yellow'

folder = os.path.join(crt_path, './ori/*')
save_patch_folder = os.path.join(crt_path, './patch')
save_rect_folder = os.path.join(crt_path, './rect')

color_tb = {}
color_tb['yellow'] = (0, 255, 255)
color_tb['green'] = (0, 255, 0)
color_tb['red'] = (0, 0, 255)
color_tb['magenta'] = (255, 0, 255)
color_tb['matlab_blue'] = (189, 114, 0)
color_tb['matlab_orange'] = (25, 83, 217)
color_tb['matlab_yellow'] = (32, 177, 237)
color_tb['matlab_purple'] = (142, 47, 126)
color_tb['matlab_green'] = (48, 172, 119)
color_tb['matlab_liblue'] = (238, 190, 77)
color_tb['matlab_brown'] = (47, 20, 162)
color = color_tb[color]
img_list = glob.glob(folder)
images = []

# make temp folder
if not os.path.exists(save_patch_folder):
    os.makedirs(save_patch_folder)
    print('mkdir [{}] ...'.format(save_patch_folder))
if not os.path.exists(save_rect_folder):
    os.makedirs(save_rect_folder)
    print('mkdir [{}] ...'.format(save_rect_folder))

for i, path in enumerate(img_list):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    base_name = os.path.splitext(os.path.basename(path))[0]
    print(i, base_name)
    # crop patch
    if img.ndim == 2:
        patch = img[h_start:h_start + h_len, w_start:w_start + w_len]
    elif img.ndim == 3:
        patch = img[h_start:h_start + h_len, w_start:w_start + w_len, :]
    else:
        raise ValueError('Wrong image dim [{:d}]'.format(img.ndim))

    # enlarge patch if necessary
    if enlarge_ratio > 1:
        H, W, _ = patch.shape
        patch = cv2.resize(patch, (W * enlarge_ratio, H * enlarge_ratio), \
            interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(save_patch_folder, base_name + '_patch.png'), patch)

    # draw rectangle
    img_rect = cv2.rectangle(img, (w_start, h_start), (w_start + w_len, h_start + h_len),
        color, line_width)
    cv2.imwrite(os.path.join(save_rect_folder, base_name + '_rect.png'), img_rect)
