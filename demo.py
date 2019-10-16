# -*- coding: utf-8 -*-
from model import CGAN
from denoise import *
import os
from bert_client import BertClientQuery
from style_transfer import Stylizer
import datetime
import time
import platform
import subprocess


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


def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


if __name__ == '__main__':
    bcq = BertClientQuery(words, topk=10)

    cgan = CGAN()
    cgan.reload('./ckpt')
    stylizer = Stylizer()

    while True:
        query = input('Enter the dish name: ')
        topk_idx = bcq.query(query)

        time_now = str(int(time.time()))
        charactor_file = './result/%s.png' % time_now
        resized_file = './result/%s_convert.png' % time_now
        styled_file = './result/%s_stylized.png' % time_now
        style_img_file = './style_image/style01.jpg'

        try:
            # generate a character
            print('generating the character ......')
            cgan.generate_one_image(topk_idx[5:], result_path=charactor_file)

            # resize and denoise
            print('resizing and denoising the character ......')
            resize_and_denoise(charactor_file, (1200, 1200), resized_file)

            # change style_image_path to generate different style!
            # if something wrong with style_image_path, it will use default style image
            print('applying style transfer ......')
            stylizer.transfer(style_image_path=style_img_file,
                                   content_image_path=resized_file,
                                   output_size=1200,
                                   result_path=styled_file
                                   )
            print('Done! Please check the files in the result folder')

            # open the generated files
            open_file(resized_file)
            open_file(styled_file)

        except Exception as e:
                        print(e)
