from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imageio
import tensorflow as tf
from tensorflow.keras import layers
from utils import *
import pathlib
import random


# Training Flags (hyperparameter configuration)
train_dir = './ckpt'

max_epochs = 100
save_model_epochs = 10
print_steps = 200
save_images_epochs = 1
batch_size = 32
learning_rate_D = 1e-4
learning_rate_G = 1e-3
k = 1 # the number of step of learning D before learning G (Not used in this code)
num_classes = 100 # number of classes for Calligraphy(should be 100, but 12 folder disappear)
num_examples_to_generate = 100
noise_dim = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE


class Generator(tf.keras.Model):
    """Build a generator that maps latent space to real space given conditions.
      G(z, c): (z, c) -> x
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = ConvTranspose(32, 3, padding='valid')
        self.conv2 = ConvTranspose(32, 3, padding='valid')
        self.conv3 = ConvTranspose(32, 4)
        self.conv4 = ConvTranspose(16, 4)
        self.conv5 = ConvTranspose(16, 4)
        self.conv6 = ConvTranspose(16, 4)
        self.conv7 = ConvTranspose(1, 4, apply_batchnorm=False, activation='tanh')

    def call(self, noise_inputs, conditions, training=True):
        """Run the model."""
        # noise_inputs: [1, 1, 100]
        # conditions: [1, 1, 100] (for Calligraphy)
        # inputs = 1 x 1 x (100 + 100) dim
        inputs = tf.concat([noise_inputs, conditions], axis=3)
        conv1 = self.conv1(inputs, training=training)  # conv1: [3, 3, 64]
        conv2 = self.conv2(conv1, training=training)  # conv2: [7, 7, 64]
        conv3 = self.conv3(conv2, training=training)  # conv3: [14, 14, 64]
        conv4 = self.conv4(conv3, training=training)  # conv3: [28, 28, 32]
        conv5 = self.conv5(conv4, training=training)  # conv3: [56, 56, 32]
        conv6 = self.conv6(conv5, training=training)  # conv3: [112, 112, 32]
        generated_images = self.conv7(conv6, training=training)  # generated_images: [224, 224, 1]

        return generated_images


class Discriminator(tf.keras.Model):
    """Build a discriminator that discriminate tuple (x, c) whether real or fake.
      D(x, c): (x, c) -> [0, 1]
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv(16, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv2 = Conv(16, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv3 = Conv(16, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv4 = Conv(32, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv5 = Conv(32, 4, 2, activation='leaky_relu')
        self.conv6 = Conv(32, 3, 2, padding='valid', activation='leaky_relu')
        self.conv7 = Conv(1, 3, 1, padding='valid', apply_batchnorm=False, activation='none')

    def call(self, image_inputs, conditions, training=True):
        """Run the model."""
        # image_inputs: [224, 224, 1]
        # conditions: 100 dim (for Calligraphy)
        # inputs: [224, 224, (1 + 100)]
        inputs = tf.concat([image_inputs,
                            conditions * tf.ones([image_inputs.shape[0],
                                                  224, 224,
                                                  num_classes])], axis=3)
        conv1 = self.conv1(inputs)  # conv1: [112, 112, 32]
        conv2 = self.conv2(conv1)  # conv1: [56, 56, 32]
        conv3 = self.conv3(conv2)  # conv1: [28, 28, 32]
        conv4 = self.conv4(conv3)  # conv4: [14, 14, 64]
        conv5 = self.conv5(conv4)  # conv5: [7, 7, 64]
        conv6 = self.conv6(conv5)  # conv6: [3, 3, 64]
        conv7 = self.conv7(conv6)  # conv7: [1, 1, 1]
        discriminator_logits = tf.squeeze(conv7, axis=[1, 2])  # discriminator_logits: [1,]

        return discriminator_logits


class CGAN:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        # discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate_D, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate_D)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate_G, beta_1=0.5)

        checkpoint_dir = train_dir
        if not tf.io.gfile.exists(checkpoint_dir):
            tf.io.gfile.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

    def _load_data(self, data_root):
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        # random.shuffle(all_image_paths)

        self.image_count = len(all_image_paths)

        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

        ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
        image_label_ds = ds.map(self._load_and_preprocess_from_path_label)
        ds = image_label_ds.shuffle(buffer_size=self.image_count)
        ds = ds.batch(batch_size)
        self.ds = ds.prefetch(buffer_size=AUTOTUNE)

    def _preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [224, 224])
        # image /= 255.0  # normalize to [0,1] range
        image = (image - 127.5) / 127.5  # normalize to [-1,1] range

        return image

    def _load_and_preprocess_image(self, path):
        image = tf.io.read_file(path)
        return self._preprocess_image(image)

    def _load_and_preprocess_from_path_label(self, path, label):
        label = tf.one_hot(label, depth=num_classes)
        label = tf.reshape(label, shape=[1, 1, num_classes])
        label = tf.cast(label, dtype=tf.float32)
        return self._load_and_preprocess_image(path), label

    def GANLoss(self, logits, is_real=True):
        """Computes standard GAN loss between `logits` and `labels`.

        Args:
            logits (`2-rank Tensor`): logits.
            is_real (`bool`): True means `1` labeling, False means `0` labeling.

        Returns:
            loss (`0-rank Tensor`): the standard GAN loss value. (binary_cross_entropy)
        """
        if is_real:
            labels = tf.ones_like(logits)
        else:
            labels = tf.zeros_like(logits)

        bce = tf.losses.BinaryCrossentropy(from_logits=True)

        #return tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        return bce(labels, logits)

    def discriminator_loss(self, real_logits, fake_logits):
        # losses of real with label "1"
        real_loss = self.GANLoss(logits=real_logits, is_real=True)
        # losses of fake with label "0"
        fake_loss = self.GANLoss(logits=fake_logits, is_real=False)

        return real_loss + fake_loss

    def generator_loss(self, fake_logits):
        # losses of Generator with label "1" that used to fool the Discriminator
        return self.GANLoss(logits=fake_logits, is_real=True)

    def train_step(self, images, labels):
        # generating noise from a uniform distribution
        noise = tf.random.uniform([len(labels), 1, 1, noise_dim], minval=-1.0, maxval=1.0)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, labels, training=True)

            real_logits = self.discriminator(images, labels, training=True)
            fake_logits = self.discriminator(generated_images, labels, training=True)

            gen_loss = self.generator_loss(fake_logits)
            disc_loss = self.discriminator_loss(real_logits, fake_logits)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss, disc_loss

    def train(self):
        const_random_vector_for_saving = tf.random.uniform([num_examples_to_generate, 1, 1, noise_dim],
                                                           minval=-1.0, maxval=1.0)
        print('Start Training.')
        num_batches_per_epoch = int(self.image_count / batch_size)
        global_step = tf.Variable(0, trainable=False)

        sample_condition = tf.eye(num_classes)
        sample_condition = tf.reshape(sample_condition, [-1, 1, 1, num_classes])
        sample_condition = sample_condition[:num_examples_to_generate]

        for epoch in range(max_epochs):

            for step, (images, labels) in enumerate(self.ds):
                start_time = time.time()

                gen_loss, disc_loss = self.train_step(images, labels)
                global_step.assign_add(1)

                if global_step.numpy() % print_steps == 0:
                    epochs = epoch + step / float(num_batches_per_epoch)
                    duration = time.time() - start_time
                    examples_per_sec = batch_size / float(duration)
                    print(
                        "Epochs: {:.2f} global_step: {} loss_D: {:.3g} loss_G: {:.3g} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
                            epochs, global_step.numpy(), disc_loss, gen_loss, examples_per_sec, duration))
                    noise = tf.random.uniform([num_examples_to_generate, 1, 1, noise_dim], minval=-1.0, maxval=1.0)
                    sample_images = self.generator(noise, sample_condition, training=False)
                    print_or_save_sample_images(sample_images.numpy(), num_examples_to_generate, is_square=True)

            if (epoch + 1) % save_images_epochs == 0:
                print("This images are saved at {} epoch".format(epoch + 1))
                sample_images = self.generator(const_random_vector_for_saving, sample_condition, training=False)
                print_or_save_sample_images(sample_images.numpy(), num_examples_to_generate,
                                            is_square=True, is_save=True, epoch=epoch + 1,
                                            checkpoint_dir=checkpoint_dir)

            # saving (checkpoint) the model every save_epochs
            if (epoch + 1) % save_model_epochs == 0:
                self.checkpoint.save(file_prefix='ckpt')

        print('Training Done.')

    def reload(self, checkpoint_dir):
        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def generate_one_image(self, vector, result_path):
        sample_condition = np.zeros(100).tolist()
        try:
            for j in vector:
                sample_condition[j] = 1.
        except Exception as e:
            print(e)

        sample_condition = np.array(sample_condition, dtype=np.float32)
        sample_condition = tf.reshape(sample_condition, [1, 1, 1, num_classes])

        const_random_vector_for_saving = tf.random.uniform([1, 1, 1, noise_dim],
                                                           minval=-1.0, maxval=1.0)

        sample_images = self.generator(const_random_vector_for_saving, sample_condition, training=False)
        save_one_sample_image(sample_images.numpy(), result_path=result_path)

    def generate_gif(self, vector, result_dir):
        sample_condition = np.zeros(100).tolist()
        const_random_vector_for_saving = tf.random.uniform([1, 1, 1, noise_dim],
                                                           minval=-1.0, maxval=1.0)

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        for j in vector:
            try:
                sample_condition[j] = 1.
            except Exception as e:
                sample_condition[0] = 1.

        for j in range(len(vector), 4, -1):
            try:
                sample_condition[vector[j]] = 0.
            except Exception as e:
                sample_condition[0] = 0.

            sample_condition_ = np.array(sample_condition, dtype=np.float32)
            sample_condition_ = tf.reshape(sample_condition_, [1, 1, 1, num_classes])
            sample_images = self.generator(const_random_vector_for_saving, sample_condition_, training=False)
            save_one_sample_image(sample_images.numpy(), result_path=os.path.join(result_dir, '%d.png' % j))

        # with imageio.get_writer(os.path.join(result_dir, 'result.gif'), mode='I') as writer:
        #     last = -1
        #     for i, j in enumerate(range(len(vector), 4, -1)):
        #         frame = 2 * (i**0.5)
        #         if round(frame) > round(last):
        #             last = frame
        #         else:
        #             continue
        #
        #         image = imageio.imread(os.path.join(result_dir, '%d.png' % j))
        #         writer.append_data(image)
        #
        #     last = -1
        #     for i, j in enumerate(range(5, len(vector))):
        #         frame = 2 * (i**0.5)
        #         if round(frame) > round(last):
        #             last = frame
        #         else:
        #             continue
        #
        #         image = imageio.imread(os.path.join(result_dir, '%d.png' % j))
        #         writer.append_data(image)
