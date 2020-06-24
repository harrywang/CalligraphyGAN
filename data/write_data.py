import os
import h5py
from PIL import Image
import numpy as np
import re
import matplotlib.pyplot as plt
from embedding import traditional2simple
from argparse import ArgumentParser


def get_all_image(path):
    image_list = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            image_list.append(item_path)
        else:
            image_list.extend(get_all_image(item_path))

    return image_list


def make_squre(img, target_size=140, color_mode='L', fill_color=255):
    x, y = img.size
    size = max(target_size, x, y)
    new_img = Image.new(color_mode, (size, size), fill_color)
    new_img.paste(img, (int((size-x) / 2), int((size - y) / 2)))
    return new_img.resize((target_size, target_size))


def get_embedding(character_csv, writer_csv):
    characters = {}
    writers = {}

    with open(writer_csv, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            writers[line.split(',')[0]] = int(line.split(',')[1])

    with open(character_csv, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            characters[line.split(',')[0]] = int(line.split(',')[1])

    return characters, writers


def data_write_all(path, label_path, output_path):
    character_label = os.path.join(label_path, 'label_character_all.csv')
    writer_label = os.path.join(label_path, 'label_character_all.csv')
    characters_embedding, writers_embedding = get_embedding(character_label, writer_label)

    print('getting image list...')
    img_path_list = get_all_image(path)

    # write image list to txt, so no need to walk through image directory again next time
    # with open('test.txt', 'w') as f:
    #     for item in img_path_list:
    #         f.writelines(item + '\n')

    image_list = np.zeros((len(img_path_list), 140, 140), dtype='uint8')
    label_character_list = np.zeros((len(img_path_list), 1), dtype='int')
    label_writer_list = np.zeros((len(img_path_list), 1), dtype='int')

    for i, item in enumerate(img_path_list):
        try:
            temp_img = Image.open(item)
            temp_img = make_squre(temp_img)
            temp_img = np.array(temp_img)
            image_list[i] = temp_img

            label_character = re.split('/|\\\\', item)[-2]
            label_character = traditional2simple(label_character)
            label_writer = re.split('/|\\\\', item)[-3]
            label_writer = traditional2simple(label_writer)
            label_character_list[i] = characters_embedding[label_character]
            label_writer_list[i] = writers_embedding[label_writer]
        except Exception as e:
            print('%s: %s' % (item, e))

        if i % 10 == 0:
            print('%d / %d' % (i, len(image_list)))

    print('writing result to hdf5......')
    try:

        f = h5py.File(os.path.join(output_path, 'chenzhongjian_all.hdf5'), 'w')
        f.create_dataset(name="image", shape=(len(img_path_list), 140, 140), data=image_list)
        f.create_dataset(name='character_label', shape=(len(img_path_list),), data=label_character_list)
        f.create_dataset(name='writer_label', shape=(len(img_path_list),), data=label_writer_list)
    except Exception as e:
        print(e)


def data_loader(data_path):
    f = h5py.File(data_path, 'r')
    data_images = f['image']
    data_character = f['character_label']
    data_writer = f['writer_label']

    return data_images, data_character, data_writer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', help='Directory of dataset', type=str, required=True)
    parser.add_argument('-l', '--label_dir', dest='label_dir', help='Directory of label csv', type=str, required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='Directory of output', type=str, required=True)

    options = parser.parse_args()

    data_write_all(options.data_dir, options.label_dir, options.output_dir)
