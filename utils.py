from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from skimage.transform import resize
import cv2
from sklearn.cluster import KMeans
import utils
import cv2


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


styles = {
    'Dekooning': './style_image/dekooning.jpg',
    'Picasso': './style_image/picasso.jpg',
    'Pollock': './style_image/pollock.jpg',
    'Rousseau': './style_image/rousseau.jpg',
    'Rothko': './style_image/rothko.jpg'
}

style_image_path = './style_image'


def get_style_dict():
    result = {}
    for item in os.listdir(style_image_path):
        result[item.split('.')[0]] = os.path.join(style_image_path, item)

    return result


# rewrite cv2 imread to read files with Chinese characters
def cv_read_img_BGR(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# rewrite cv2 imwrite to write files with Chinese characters
def cv_write_img_BGR(img, result_path):
    cv2.imencode('.png', img)[1].tofile(result_path)


# function to get topk color in image
def color_cluster(image_path, topk=5):
    """
    get top-k colors in an image
    :param topk:
    :param image_path:
    :return:
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # cluster the pixel intensities
    clt = KMeans(n_clusters=topk)
    clt.fit(image)

    # below is the code to visualize the resultsv
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)
    #
    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        # return the histogram
        return hist

    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    return bar, clt.cluster_centers_
    # # show our color bart
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()


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
