import imageio
import os.path
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

root = '/home/xtwang/Projects/BasicSR/codes/scripts/'
save_dir = './tmp/'
image_names = [
    'baby.png'
    , 'baby_x2.png'
    , 'baby_x4.png'
]
text_lists = [
    'baby'
    , 'baby_x2'
    , 'baby_x4'
]
h_start, h_len = 10, 400
w_start, w_len = 10, 400
enlarge = 1
txt_pos = (1, 10)  # w, h
front_size = 30
duration = 0.8  # second

images = []
font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", front_size)

# make temp dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('mkdir ... ' + save_dir)

for filename, write_txt in zip(image_names, text_lists):
    img = imageio.imread(os.path.join(root, filename))
    # crop image
    if img.ndim == 2:
        img = img[h_start:h_start + h_len, w_start:w_start + w_len]
    elif img.ndim == 3:
        img = img[h_start:h_start + h_len, w_start:w_start + w_len, :]
    else:
        raise ValueError('Wrong image dim [%d]' % img.ndim)
    # enlarge image if necessary
    img_pil = Image.fromarray(img).convert('RGBA')
    W, H = img_pil.size
    img_pil = img_pil.resize((W * enlarge, H * enlarge), Image.BICUBIC)
    # add text
    txt = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt)
    d.text(txt_pos, write_txt, font=font, fill=(255, 0, 0, 255))

    out_pil = Image.alpha_composite(img_pil, txt).convert('RGB')
    out = np.array(out_pil)
    # make gif
    images.append(out)
    imageio.imwrite(os.path.join(save_dir, filename), out)
imageio.mimsave(os.path.join(save_dir, 'out.gif'), images, format='GIF', duration=duration)
