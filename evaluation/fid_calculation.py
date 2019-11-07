from evaluation.util import calculate_fid, read_images, generate_target_word
import os
from PIL import Image
from keras.applications.inception_v3 import InceptionV3


if __name__ == '__main__':
    word_list = ['且', '九', '和', '元', '正']
    for item in word_list:
        generate_target_word(item, checkpoint_dir='../ckpt', result_dir='./' + item, size=5)

    # prepare the inception v3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # To generate a FID matrix
    # word_list_1 is for column
    # word_List_2 is for row
    word_list_1 = word_list
    word_list_2 = word_list

    fid_matrix = [[0.] * len(word_list_2) for i in range(0, len(word_list_1))]

    path_1 = './'
    path_2 = './'

    for row, item in enumerate(word_list_1):
        data_path_1 = os.path.join(path_1, item)

        for col, i in enumerate(word_list_2):
            data_path_2 = os.path.join(path_2, i)

            images1 = read_images(data_path_1)
            images2 = read_images(data_path_2)

            fid = calculate_fid(model, images1, images2)

            print(i, fid)
            fid_matrix[row][col] = fid
