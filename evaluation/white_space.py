import numpy as np
from PIL import Image


def white_space_ratio(image_path, threshold=240):
    white_space_cnt = 0.

    img = np.array(Image.open(image_path).convert('L'))

    width, height = img.shape[0], img.shape[1]

    for i in range(width):
        for j in range(height):
            if img[i][j] >= threshold:
                white_space_cnt += 1

    return white_space_cnt / (width * height)
