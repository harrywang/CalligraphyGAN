import numpy as np
import cv2
from PIL import Image


def denoise_by_value(img, threshold):
    img = np.array(img).tolist()

    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > threshold:
                img[i][j] = 255.

    return np.array(img, dtype='uint8')


def resize_and_denoise(img, result_size, denoise_th):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, result_size, interpolation=cv2.INTER_AREA)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 100, 100, 11, 27)
    result_resized = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    result_denoised = denoise_by_value(dst, denoise_th)
    result_denoised = cv2.cvtColor(result_denoised, cv2.COLOR_GRAY2RGB)

    return result_resized, result_denoised


def resize(img_path, result_size, dst_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, result_size, interpolation=cv2.INTER_AREA)
    new_img = Image.fromarray(np.array(img, dtype=np.uint8))
    new_img.save(dst_path)
