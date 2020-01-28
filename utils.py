from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob

import numpy as np
from numpy import cov
import matplotlib.pyplot as plt

import PIL
import imageio
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from PIL import Image
from skimage.transform import resize
from scipy.linalg import sqrtm

import cv2


# rewrite cv2 imread to read files with Chinese characters
def cv_read_img_BGR(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# rewrite cv2 imwrite to write files with Chinese characters
def cv_write_img_BGR(img, result_path):
    cv2.imencode('.png', img)[1].tofile(result_path)


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def save_one_sample_image(sample_images, result_path, is_square=True):
    if len(sample_images.shape) == 2:
        size = int(np.sqrt(sample_images.shape[1]))
    elif len(sample_images.shape) > 2:
        size = sample_images.shape[1]
        channel = sample_images.shape[3]
    else:
        raise ValueError('Not valid a shape of sample_images')

    if not is_square:
        print_images = sample_images[:1, ...]
        print_images = print_images.reshape([1, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, 1 * size, channel])
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(1, 1))
        plt.imshow(print_images * 0.5 + 0.5)  # , cmap='gray')
        plt.axis('off')

    else:
        num_columns = int(np.sqrt(1))
        max_print_size = int(num_columns ** 2)
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        print_images = [print_images[:, i * size * num_columns:(i + 1) * size * num_columns] for i in
                        range(num_columns)]
        print_images = np.concatenate(tuple(print_images), axis=0)
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(num_columns, num_columns))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.imshow(print_images * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    plt.savefig(result_path, dpi=300)
    plt.close()


def print_or_save_sample_images(sample_images, max_print_size=25,
                                is_square=False, is_save=False, epoch=None,
                                checkpoint_dir='./train'):
    available_print_size = list(range(1, 101))
    assert max_print_size in available_print_size
    if len(sample_images.shape) == 2:
        size = int(np.sqrt(sample_images.shape[1]))
    elif len(sample_images.shape) > 2:
        size = sample_images.shape[1]
        channel = sample_images.shape[3]
    else:
        ValueError('Not valid a shape of sample_images')

    if not is_square:
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(max_print_size, 1))
        plt.imshow(print_images * 0.5 + 0.5)  # , cmap='gray')
        plt.axis('off')

    else:
        num_columns = int(np.sqrt(max_print_size))
        max_print_size = int(num_columns ** 2)
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        print_images = [print_images[:, i * size * num_columns:(i + 1) * size * num_columns] for i in
                        range(num_columns)]
        print_images = np.concatenate(tuple(print_images), axis=0)
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(num_columns, num_columns))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.imshow(print_images * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    if is_save and epoch is not None:
        filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


# TODO: rewrite this function
def generate_gif(vector, result_dir):
    with imageio.get_writer(os.path.join(result_dir, 'result.gif'), mode='I') as writer:
        for j in range(0, 40):

            image = imageio.imread(os.path.join(result_dir, '%d.png' % j))
            writer.append_data(image)

        for j in range(0, 10):
            image = imageio.imread(os.path.join(result_dir, '39.png'))
            writer.append_data(image)

        for j in range(39, 0, -1):
            image = imageio.imread(os.path.join(result_dir, '%d.png' % j))
            writer.append_data(image)
