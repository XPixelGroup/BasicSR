import cv2
import os
# import glob

current_dir = os.getcwd()
# print(current_dir)
files = os.listdir(current_dir + '/gt')
files = [f for f in files if os.path.isfile(os.path.join(current_dir + '/gt/', f))]
# print(files)
# print(files)
for f in files:
    img = cv2.imread(current_dir + '/gt/' + f)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(current_dir + '/gt/' + f, img_gray)
    # img_resize.save(current_dir + '/lq_x4/' + f)
    print(f+' done')
    # print(f, img_resize.size)

# lq_path = '/lq'
# img = Image.open('/Users/maruyan/desktop/鈴木研/SR_experiments/dataset/lq/beam_test (1).png')
# print(img.size)
