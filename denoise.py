import numpy as np
import cv2
from PIL import Image


def denoise_by_value(threshold, img_path, result_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = np.array(img).tolist()

    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > threshold:
                img[i][j] = 255.

    new_img = Image.fromarray(np.array(img, dtype=np.uint8))
    new_img.save(result_path)


def resize_and_denoise(img_path, result_size, dst_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, result_size, interpolation=cv2.INTER_AREA)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 100, 100, 11, 27)
    cv2.imwrite(dst_path, dst)
    denoise_by_value(255./2, dst_path, dst_path)
