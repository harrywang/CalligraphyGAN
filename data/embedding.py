"""
Embedding the writers and characters for future processing
"""
import os
from langconv import *
from argparse import ArgumentParser


def traditional2simple(s):
    return Converter('zh-hans').convert(s)


def generate_embedding_all(path, character_path, writer_path):
    writers = {}
    characters = {}

    s_path = os.path.join(path, 'simple')
    t_path = os.path.join(path, 'traditional')

    for item in os.listdir(s_path):
        if item not in writers.keys():
            writers[item] = len(writers.keys())
        temp_path = os.path.join(s_path, item)
        for character in os.listdir(temp_path):
            if character not in characters.keys():
                characters[character] = len(characters.keys())

    for item in os.listdir(t_path):
        temp_item = traditional2simple(item)
        if temp_item not in writers.keys():
            writers[temp_item] = len(writers.keys())
        temp_path = os.path.join(t_path, item)
        for character in os.listdir(temp_path):
            temp_character = traditional2simple(character)
            if temp_character not in characters.keys():
                characters[temp_character] = len(characters.keys())

    with open(character_path, 'w', encoding='utf-8') as f:
        for item in characters.keys():
            f.write('%s,%d\n' % (item, characters[item]))

    with open(writer_path, 'w', encoding='utf-8') as f:
        for item in writers.keys():
            f.write('%s,%d\n' % (item, writers[item]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', dest='data_dir', help='Path of Dataset', type=str, required=True)
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='Path of Output csv', type=str, required=True)

    options = parser.parse_args()

    data_path = options.data_dir
    generate_embedding_all(path=data_path,
                           character_path=os.path.join(options.output_dir, '/label_character_all.csv'),
                           writer_path=os.path.join(options.output_path, './label_writer_all.csv')
                           )
