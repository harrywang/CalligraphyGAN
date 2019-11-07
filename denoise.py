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
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, result_size, interpolation=cv2.INTER_AREA)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 100, 100, 11, 27)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    cv2.imencode('.png', dst)[1].tofile(dst_path)
    # cv2.imwrite(dst_path, dst)
    denoise_by_value(255./2, dst_path, dst_path)


def resize(img_path, result_size, dst_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, result_size, interpolation=cv2.INTER_AREA)
    new_img = Image.fromarray(np.array(img, dtype=np.uint8))
    new_img.save(dst_path)
# =======
# if __name__ == '__main__':
#     name = '三文鱼柳配藜麦、番茄和青酱'
#     resize_and_denoise('./1020/%s/1571284852.png' % name.encode('utf8'), (1200, 1200), './1020/%s/1571284852_convert.png' % name.encode('utf8'))
# >>>>>>> Stashed changes
