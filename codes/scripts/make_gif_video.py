"""
Add text to images, then make gif/video sequence from images.

Since the created gif has low quality with color issues, use this script to generate image with
text and then use `gifski`.

Call `ffmpeg` to make video.
"""

import os.path
import numpy as np
import cv2


crt_path = os.path.dirname(os.path.realpath(__file__))

# configurations
img_name_list = ['x1', 'x2', 'x3', 'x4', 'x5']
ext = '.png'
text_list = ['1', '2', '3', '4', '5']
h_start, h_len = 0, 576
w_start, w_len = 10, 352
enlarge_ratio = 1
txt_pos = (10, 50)  # w, h
font_size = 1.5
font_thickness = 4
color = 'red'
duration = 0.8  # second
use_imageio = False  # use imageio to make gif
make_video = False  # make video using ffmpeg

is_crop = True
if h_start == 0 or w_start == 0:
    is_crop = False # do not crop

img_name_list = [x + ext for x in img_name_list]
input_folder = os.path.join(crt_path, './ori')
save_folder = os.path.join(crt_path, './ori')
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

img_list = []

# make temp dir
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print('mkdir [{}] ...'.format(save_folder))
if make_video:
    # tmp folder to save images for video
    tmp_video_folder = os.path.join(crt_path, '_tmp_video')
    if not os.path.exists(tmp_video_folder):
        os.makedirs(tmp_video_folder)

idx = 0
for img_name, write_txt in zip(img_name_list, text_list):
    img = cv2.imread(os.path.join(input_folder, img_name), cv2.IMREAD_UNCHANGED)
    base_name = os.path.splitext(img_name)[0]
    print(base_name)
    # crop image
    if is_crop:
        print('Crop image ...')
        if img.ndim == 2:
            img = img[h_start:h_start + h_len, w_start:w_start + w_len]
        elif img.ndim == 3:
            img = img[h_start:h_start + h_len, w_start:w_start + w_len, :]
        else:
            raise ValueError('Wrong image dim [{:d}]'.format(img.ndim))

    # enlarge img if necessary
    if enlarge_ratio > 1:
        H, W, _ = img.shape
        img = cv2.resize(img, (W * enlarge_ratio, H * enlarge_ratio), \
            interpolation=cv2.INTER_CUBIC)

    # add text
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, write_txt, txt_pos, font, font_size, color, font_thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_folder, base_name + '_text.png'), img)
    if make_video:
        idx += 1
        cv2.imwrite(os.path.join(tmp_video_folder, '{:05d}.png'.format(idx)), img)

    img = np.ascontiguousarray(img[:, :, [2, 1, 0]])
    img_list.append(img)

if use_imageio:
    import imageio
    imageio.mimsave(os.path.join(save_folder, 'out.gif'), img_list, format='GIF', duration=duration)

if make_video:
    os.system('ffmpeg -r {:f} -i {:s}/%05d.png -vcodec mpeg4 -y {:s}/movie.mp4'.format(
        1 / duration, tmp_video_folder, save_folder))

if os.path.exists(tmp_video_folder):
    os.system('rm -rf {}'.format(tmp_video_folder))
