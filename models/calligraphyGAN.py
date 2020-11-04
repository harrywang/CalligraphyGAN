import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers as layers
from models.calligraphyGAN_config import config


class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding='same',
               activation='relu', apply_batchnorm=True, norm_momentum=0.9, norm_epsilon=1e-5,
               leaky_relu_alpha=0.2):
        super(Conv, self).__init__()
        assert activation in ['relu', 'leaky_relu', 'none']
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm
        self.leaky_relu_alpha = leaky_relu_alpha

        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=(kernel_size, kernel_size),
                                  strides=strides,
                                  padding=padding,
                                  kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                  use_bias=not self.apply_batchnorm)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization(momentum=norm_momentum,
                                                     epsilon=norm_epsilon)

    def call(self, x, training=True):
        # convolution
        x = self.conv(x)

        # batchnorm
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)

        # activation
        if self.activation == 'relu':
            x = tf.nn.relu(x)
        elif self.activation == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)
        else:
            pass

        return x


class ConvTranspose(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=2, padding='same',
               activation='relu', apply_batchnorm=True, norm_momentum=0.9, norm_epsilon=1e-5):
        super(ConvTranspose, self).__init__()
        assert activation in ['relu', 'sigmoid', 'tanh', 'none']
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm

        self.up_conv = layers.Conv2DTranspose(filters=filters,
                                              kernel_size=(kernel_size, kernel_size),
                                              strides=strides,
                                              padding=padding,
                                              kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                              use_bias=not self.apply_batchnorm)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization(momentum=norm_momentum,
                                                     epsilon=norm_epsilon)

    def call(self, x, training=True):
        # conv transpose
        x = self.up_conv(x)

        # batchnorm
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)

        # activation
        if self.activation == 'relu':
            x = tf.nn.relu(x)
        elif self.activation == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif self.activation == 'tanh':
            x = tf.nn.tanh(x)
        else:
            pass

        return x


class Dense(tf.keras.Model):
    def __init__(self, units, activation='relu', apply_batchnorm=True, norm_momentum=0.9, norm_epsilon=1e-5,
               leaky_relu_alpha=0.2):
        super(Dense, self).__init__()
        assert activation in ['relu', 'leaky_relu', 'none']
        self.activation = activation
        self.apply_batchnorm = apply_batchnorm
        self.leaky_relu_alpha = leaky_relu_alpha

        self.dense = layers.Dense(units=units,
                                  kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                  use_bias=not self.apply_batchnorm)
        if self.apply_batchnorm:
            self.batchnorm = layers.BatchNormalization(momentum=norm_momentum,
                                                     epsilon=norm_epsilon)

    def call(self, x, training=True):
        # dense
        x = self.dense(x)

        # batchnorm
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)

        # activation
        if self.activation == 'relu':
            x = tf.nn.relu(x)
        elif self.activation == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=self.leaky_relu_alpha)
        else:
            pass

        return x


class ClassEmbedding(keras.Model):
    """
    Convert input class (one-hot) into a vector with the same size as z

    Because there are too many characters in our dataset (we use 1000 characters),
    using one-hot embedding to represent each character will waste space and time.
    """
    def __init__(self):
        super(ClassEmbedding, self).__init__()
        self.embed = Dense(config.class_embedding_dim, apply_batchnorm=False, activation='none')

    def call(self, x, training=True):
        embed_class = self.embed(x, training=training)

        return embed_class


class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = ConvTranspose(32, 3, padding='valid')
        self.conv2 = ConvTranspose(32, 3, padding='valid')
        self.conv3 = ConvTranspose(32, 4)
        self.conv4 = ConvTranspose(16, 4)
        self.conv5 = ConvTranspose(16, 4)
        self.conv6 = ConvTranspose(16, 4)
        self.conv7 = ConvTranspose(1, 4, apply_batchnorm=False, activation='tanh')

    # def call(self, noise_inputs, class_embed, training=True):
    def call(self, noise_inputs, class_embed, training=True):
        # noise_inputs: [?, 1, 1, config.z_dim]
        # class_embed:  [?, 1, 1, config.class_embedding_dim]
        inputs = tf.concat([noise_inputs, class_embed], axis=3)
        conv1 = self.conv1(inputs, training=training)  # conv1: [3, 3, 64]
        conv2 = self.conv2(conv1, training=training)  # conv2: [7, 7, 64]
        conv3 = self.conv3(conv2, training=training)  # conv3: [14, 14, 64]
        conv4 = self.conv4(conv3, training=training)  # conv3: [28, 28, 32]
        conv5 = self.conv5(conv4, training=training)  # conv3: [56, 56, 32]
        conv6 = self.conv6(conv5, training=training)  # conv3: [112, 112, 32]
        generated_images = self.conv7(conv6, training=training)  # generated_images: [224, 224, 1]

        return generated_images

    def model(self):
        x = layers.Input(shape=(1, 1, config.noise_dim))
        y = layers.Input(shape=(1, 1, config.class_embedding_dim))
        return keras.Model(inputs=[x, y], outputs=self.call(x, y))


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv(16, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv2 = Conv(16, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv3 = Conv(16, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv4 = Conv(32, 4, 2, apply_batchnorm=False, activation='leaky_relu')
        self.conv5 = Conv(32, 4, 2, activation='leaky_relu')
        self.conv6 = Conv(32, 3, 2, padding='valid', activation='leaky_relu')
        self.conv7 = Conv(1, 3, 1, padding='valid', apply_batchnorm=False, activation='none')

    def call(self, image_inputs, class_embed, training=True):
        # image_inputs: [?, config.image_size, config.image_size, 1]
        # class_embed:  [?, 1, 1, config.class_embedding_dim]
        # inputs:       [?, config.image_size, config.image_size, (1 + config.class_embedding_dim)]
        inputs = tf.concat(
            [image_inputs,
             class_embed *
             tf.ones([image_inputs.shape[0], config.image_size, config.image_size, config.class_embedding_dim])
             ],
            axis=3)

        conv1 = self.conv1(inputs)  # conv1: [112, 112, 32]
        conv2 = self.conv2(conv1)  # conv1: [56, 56, 32]
        conv3 = self.conv3(conv2)  # conv1: [28, 28, 32]
        conv4 = self.conv4(conv3)  # conv4: [14, 14, 64]
        conv5 = self.conv5(conv4)  # conv5: [7, 7, 64]
        conv6 = self.conv6(conv5)  # conv6: [3, 3, 64]
        conv7 = self.conv7(conv6)  # conv7: [1, 1, 1]
        discriminator_logits = tf.squeeze(conv7, axis=[1, 2])  # discriminator_logits: [1,]

        return discriminator_logits

    def model(self):
        x = layers.Input(shape=(224, 224, 1))
        y = layers.Input(shape=(1, 1, config.class_embedding_dim))
        return keras.Model(inputs=[x, y], outputs=self.call(x, y))


class CalligraphyGAN(keras.Model):
    def __init__(self):
        super(CalligraphyGAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()
        self.class_embed = ClassEmbedding()

    def generate_images(self, vector, number):
        label = np.array([0] * config.class_dim)
        for idx in vector:
            label[idx] = 1
        label = np.reshape(label, (-1, 1, 1, config.class_dim))
        label_embed = self.class_embed(label, training=False)

        sample_condition = tf.concat([label_embed] * number, axis=0)

        z = tf.random.uniform([number, 1, 1, config.noise_dim], minval=-1., maxval=1.)

        images = self.generator(z, sample_condition, training=False)

        return images.numpy()

    def generate_one_image(self, topk_idx):
        label = np.array([0] * config.class_dim)
        for idx in topk_idx:
            label[idx] = 1
        label = np.reshape(label, (-1, 1, 1, config.class_dim))

        z = tf.random.uniform([1, 1, 1, config.noise_dim], minval=-1., maxval=1.)

        label_embed = self.class_embed(label, training=False)
        generated = self.generator(z, label_embed, training=False)

        return generated.numpy()
