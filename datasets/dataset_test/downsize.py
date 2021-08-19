from PIL import Image
import os
# import glob

current_dir = os.getcwd()
print(current_dir)
files = os.listdir(current_dir + '/lq')
files = [f for f in files if os.path.isfile(os.path.join(current_dir + '/lq/', f))]

for f in files:
    img = Image.open(current_dir + '/lq/' + f)
    img_resize = img.resize((int(img.width / 4), int(img.height / 4)))
    img_resize.save(current_dir + '/lq_x4/' + f)
    print(f+' done')
    print(f, img_resize.size)
