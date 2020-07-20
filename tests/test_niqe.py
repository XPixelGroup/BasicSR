import cv2

from basicsr.metrics import calculate_niqe


def main():
    img_path = 'tests/data/baboon.png'
    img = cv2.imread(img_path)
    niqe_result = calculate_niqe(img, 0, input_order='HWC', convert_to='y')
    print(niqe_result)


if __name__ == '__main__':
    main()
