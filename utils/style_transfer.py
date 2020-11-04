import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


class Stylizer:
    def __init__(self):
        # Pay attention. Bad network or GreatWall may cause failure.
        os.environ["TFHUB_CACHE_DIR"] = './ckpt/tfhub'
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        self.hub_module = hub.load(hub_handle)
        # print('done')

    # Define image loading and visualization functions  { display-mode: "form" }
    # Returns a cropped square image.
    def crop_center(self, image):
        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2
        offset_x = max(shape[2] - shape[1], 0) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, new_shape, new_shape
        )

        return image

    def load_image(self, image_url, image_size=(256, 256), preserve_aspect_ratio=True):
        """Loads and preprocesses images."""
        # Cache image file locally.
        image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
        # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
        img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
        if img.max() > 1.0:
            img = img / 255.
        if len(img.shape) == 3:
            img = tf.stack([img, img, img], axis=-1)

        img = self.crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img

    def show_n(self, images, titles=('',)):
        n = len(images)
        image_sizes = [image.shape[1] for image in images]
        w = (image_sizes[0] * 6) // 320
        plt.figure(figsize=(w * n, w))
        gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
        for i in range(n):
            plt.subplot(gs[i])
            plt.imshow(images[i][0], aspect='equal')
            plt.axis('off')
            plt.title(titles[i] if len(titles) > i else '')
        plt.show()

    def import_image(self, img, image_size=(256, 256)):
        img = img.astype(np.float32)[np.newaxis, ...]
        img = img[:, :, :, :3]
        if img.max() > 1.0:
            img = img / 255.
        if len(img.shape) == 3:
            img = tf.stack([img, img, img], axis=-1)
        img = self.crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

        return img

    def transfer(self, content_image, output_size, style_image_path, style_img_size=(256, 256)):
        # The style prediction model was trained with image size 256 and it's the
        # recommended image size for the style image (though, other sizes work as
        # well but will lead to different results).
        self.style_img_size = style_img_size  # Recommended to keep it at 256.

        try:
            img = np.array(Image.open(style_image_path))
            self.style_image = self.import_image(img, style_img_size)
        except Exception as e:
            print(e)
            style_image_url = 'https://user-images.githubusercontent.com/595772/66951165-cc179880-f027-11e9-8e46-407defc5c2a4.jpg'
            self.style_image = self.load_image(style_image_url, style_img_size)

        output_image_size = output_size

        # The content image size can be arbitrary.
        content_img_size = (output_image_size, output_image_size)

        content_image = self.import_image(content_image, content_img_size)
        style_image = tf.nn.avg_pool(self.style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

        # Stylize content image with given style image.
        # This is pretty fast within a few milliseconds on a GPU.
        outputs = self.hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        stylized_image = np.array(stylized_image)
        stylized_image = np.reshape(stylized_image, (output_size, output_size, 3))
        stylized_image *= 255

        return stylized_image.astype('uint8')
