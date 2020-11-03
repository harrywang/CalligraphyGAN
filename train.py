import tensorflow as tf
from models.calligraphyGAN import Generator, Discriminator, ClassEmbedding
from models.calligraphyGAN_config import config
from models.calligraphyGAN_dataset import CalligraphyDataset
import os
import time
import matplotlib.pyplot as plt
import numpy as np

ds = CalligraphyDataset(data_dir='./data/character-1000',
                        character_csv='./data/label_character.csv',
                        batch_size=config.batch_size,
                        shuffle=True,
                        repeat=False
                        )

generator = Generator()
discriminator = Discriminator()
class_embed = ClassEmbedding()

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


discriminator_optimizer = tf.keras.optimizers.RMSprop(config.learning_rate_D)
generator_optimizer = tf.keras.optimizers.Adam(config.learning_rate_G, beta_1=0.5)
class_embed_optimizer = tf.keras.optimizers.Adam(config.learning_rate_G, beta_1=0.5)

checkpoint_prefix = os.path.join(config.checkpoint_dir, 'train')
checkpoint = tf.train.Checkpoint(
    discriminator_optimizer=discriminator_optimizer,
    generator_optimizer=generator_optimizer,
    class_embed_optimizer=class_embed_optimizer,
    generator=generator,
    discriminator=discriminator,
    class_embed=class_embed
)
manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_dir, max_to_keep=10)


def train_step(images, labels):
    noise = tf.random.uniform([len(labels), 1, 1, config.noise_dim], minval=-1., maxval=1.)

    with tf.GradientTape(persistent=True) as tape:
        labels_embed = class_embed(labels)
        generated_images = generator(noise, labels_embed, training=True)

        real_logits = discriminator(images, labels_embed, training=True)
        fake_logits = discriminator(generated_images, labels_embed, training=True)

        gen_loss = generator_loss(fake_logits)
        disc_loss = discriminator_loss(real_logits, fake_logits)

    gen_gradients = tape.gradient(gen_loss, generator.trainable_variables)
    dis_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
    embed_gradients = tape.gradient(disc_loss, class_embed.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
    class_embed_optimizer.apply_gradients(zip(embed_gradients, class_embed.trainable_variables))

    return gen_loss, disc_loss


# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
# To visualize progress in the animated GIF
const_random_vector_for_saving = tf.random.uniform([config.num_examples_to_generate, 1, 1, config.noise_dim],
                                                   minval=-1., maxval=1.)


def print_or_save_sample_images(sample_images, max_print_size=config.num_examples_to_generate,
                                is_square=False, is_save=False, epoch=None,
                                example_dir=None):
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
        filepath = os.path.join(example_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


def train():
    global_step = tf.Variable(0, trainable=False)

    sample_condition = tf.eye(config.class_dim)
    sample_condition = tf.reshape(sample_condition, [-1, 1, 1, config.class_dim])
    sample_condition = sample_condition[:config.num_examples_to_generate]

    for epoch in range(config.max_epochs):
        start_time = time.time()
        for step, (images, labels) in enumerate(ds.dataset):
            labels = np.array([ds.characters[item.numpy().decode('utf-8')] for item in labels])
            labels = tf.one_hot(labels, depth=config.class_dim)
            labels = tf.reshape(labels, shape=[-1, 1, 1, config.class_dim])
            labels = tf.cast(labels, dtype=tf.float32)

            gen_loss, disc_loss = train_step(images, labels)
            global_step.assign_add(1)

            if global_step.numpy() & config.print_steps == 0:
                duration = time.time() - start_time
                examples_per_sec = config.batch_size / float(duration)

                print(
                    "Epochs: {:.2f} global_step: {} loss_D: {:.3g} loss_G: {:.3g} ({:.2f} examples/sec; {:.3f} sec/batch)".format(
                        epoch, global_step.numpy(), disc_loss, gen_loss, examples_per_sec, duration))

        if (epoch + 1) % config.save_images_epochs == 0:
            condition_embed = class_embed(sample_condition, training=False)
            sample_images = generator(const_random_vector_for_saving, condition_embed, training=False)
            print_or_save_sample_images(sample_images.numpy(), config.num_examples_to_generate,
                                        is_square=True, is_save=True, epoch=epoch + 1,
                                        example_dir=config.example_dir)
            print("This images are saved at {} epoch".format(epoch + 1))

        if (epoch + 1) % config.save_model_epochs == 0:
            ckpt_save_path = manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start_time))


train()
