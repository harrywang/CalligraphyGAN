import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
import os
from PIL import Image
from model import CGAN
from denoise import resize_and_denoise


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def read_images(path):
    images = []

    for file in os.listdir(path):
        images.append(np.asarray(Image.open(os.path.join(path, file)).convert('RGB')))

    images = np.array(images)
    images = scale_images(images, (299, 299, 3))
    images = preprocess_input(images)

    return images


def generate_target_word(word, checkpoint_dir='../ckpt', result_dir='./', size=10):
    words = ['且', '世', '东', '九', '亭', '今', '从', '令', '作', '使',
             '侯', '元', '光', '利', '印', '去', '受', '右', '司', '合',
             '名', '周', '命', '和', '唯', '堂', '士', '多', '夜', '奉',
             '女', '好', '始', '字', '孝', '守', '宗', '官', '定', '宜',
             '室', '家', '寒', '左', '常', '建', '徐', '御', '必', '思',
             '意', '我', '敬', '新', '易', '春', '更', '朝', '李', '来',
             '林', '正', '武', '氏', '永', '流', '海', '深', '清', '游',
             '父', '物', '玉', '用', '申', '白', '皇', '益', '福', '秋',
             '立', '章', '老', '臣', '良', '莫', '虎', '衣', '西', '起',
             '足', '身', '通', '遂', '重', '陵', '雨', '高', '黄', '鼎']

    index = 0
    for i, item in enumerate(words):
        if item == word:
            index = i
    vector = [0 for i in range(0, 100)]
    vector[index] = 1

    cgan = CGAN()
    cgan.reload(checkpoint_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for i in range(0, size):
        save_path = os.path.join(result_dir, '%d.png' % i)
        cgan.generate_one_image(vector, save_path)
        resize_and_denoise(img_path=save_path,
                           result_size=(300, 300),
                           dst_path=save_path)


def generate_target_vector(vector, checkpoint_dir='../ckpt', result_dir='./', size=10):
    cgan = CGAN()
    cgan.reload(checkpoint_dir)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    for i in range(0, size):
        save_path = os.path.join(result_dir, '%d.png' % i)
        cgan.generate_one_image(vector, save_path)
        resize_and_denoise(img_path=save_path,
                           result_size=(300, 300),
                           dst_path=save_path)
