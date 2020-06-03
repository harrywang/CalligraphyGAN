import h5py
import matplotlib.pyplot as plt


def get_embedding(writer_csv, character_csv):
    writers = {}
    characters = {}
    with open(writer_csv, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            writers[line.split(',')[0]] = int(line.split(',')[1])

    with open(character_csv, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            characters[line.split(',')[0]] = int(line.split(',')[1])

    return characters, writers


def data_loader(data_path):
    f = h5py.File(data_path, 'r')
    data_images = f['image']
    data_character = f['character_label']
    data_writer = f['writer_label']

    return data_images, data_character, data_writer


def main():
    type_suffix = 'traditional'
    data_path = './data/chenzhongjian_%s.hdf5' % type_suffix
    writer_csv = './data/label_writer_%s.csv' % type_suffix
    character_csv = './data/label_character_%s.csv' % type_suffix
    data_images, data_character, data_writer = data_loader(data_path)
    characters, writers = get_embedding(writer_csv, character_csv)
    writers = list(writers.keys())
    characters = list(characters.keys())

    # show a sample
    plt.figure()
    plt.imshow(data_images[0])
    plt.title('%s: %s'
              % (characters[int(data_character[0])],
                 writers[int(data_writer[0])]),
              fontdict={'family': 'SimHei'})
    plt.show()


if __name__ == '__main__':
    main()
