import cv2
import os

current_dir = os.getcwd()
files = os.listdir(current_dir + '/gt')
files = [f for f in files if os.path.isfile(os.path.join(current_dir + '/gt/', f))]

# for f in files:
#     img = cv2.imread(current_dir + '/gt/' + f)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(current_dir + '/gt/' + f, img_gray)
#     print(f+' done')

x = files[0]
img = cv2.imread(current_dir + '/gt/' + x, 0)
print(img.shape)
