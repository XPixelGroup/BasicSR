import cv2
import os

current_dir = os.getcwd()
files = os.listdir(current_dir + '/gt')
files = [f for f in files if os.path.isfile(os.path.join(current_dir + '/gt/', f))]

x = files[0]
img = cv2.imread(current_dir + '/gt/' + x, 0)
print(img.shape)
