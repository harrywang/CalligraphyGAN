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
import inquirer
import numpy as np
import PIL, glob
from PIL import Image
from math import ceil, floor


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
    print('initializing ......')
    bcq = BertClientQuery(words, topk=10)

    cgan = CGAN()
    cgan.reload('./ckpt')
    stylizer = Stylizer()

    while True:
        query = input("Enter dish name - [default:红烧肉卤鸡蛋]:")
        if len(query) == 0 :
            query = "红烧肉卤鸡蛋"

        topk_idx = bcq.query(query)
        print(topk_idx)

        time_now = str(int(time.time()))
        charactor_file = './result/%s.png' % time_now
        resized_file = './result/%s_convert.png' % time_now
        styled_file = './result/%s_stylized.png' % time_now
        final_output_file = './result/%s_final.jpg' % time_now


        try:
            # generate a character
            print('generating the character...')
            cgan.generate_one_image(topk_idx[5:], result_path=charactor_file)

            # resize and denoise
            print('resizing and denoising the character...')

            #resize(charactor_file, (1200, 1200), resized_file)

            # denoising will change the style transfer effect to the background
            resize_and_denoise(charactor_file, (1200, 1200), resized_file)



            # change style_image_path to generate different style!
            # if something wrong with style_image_path, it will use default style image

            styles = [
                inquirer.List('style',
                              message="Choose a style:",
                              choices=['Picasso', 'Pollock', 'Rousseau', 'Rothko', 'deKooning'],
                          ),
            ]

            chosen_style = inquirer.prompt(styles)
            style_img_file = './style_image/%s.jpg' % chosen_style['style'].lower()

            print('applying style transfer...')
            stylizer.transfer(style_image_path=style_img_file,
                                   content_image_path=resized_file,
                                   output_size=1200,
                                   result_path=styled_file
                                   )


            # combine the output images into one
            print('Finishing up...')
            frame_width = 1920
            images_per_row = 4
            padding = 2

            images = glob.glob('./result/%s*.png' % time_now)
            images.append(style_img_file)
            images.sort()
            print(images)

            img_width, img_height = Image.open(images[0]).size
            sf = (frame_width-(images_per_row-1)*padding)/(images_per_row*img_width)       #scaling factor
            scaled_img_width = ceil(img_width*sf)                   #s
            scaled_img_height = ceil(img_height*sf)

            number_of_rows = ceil(len(images)/images_per_row)
            frame_height = ceil(sf*img_height*number_of_rows)

            new_im = Image.new('RGB', (frame_width, frame_height))

            i,j=0,0
            for num, im in enumerate(images):
                if num%images_per_row==0:
                    i=0
                im = Image.open(im)
                #Here I resize my opened image, so it is no bigger than 100,100
                im.thumbnail((scaled_img_width,scaled_img_height))
                #Iterate through a 4 by 4 grid with 100 spacing, to place my image
                y_cord = (j//images_per_row)*scaled_img_height
                new_im.paste(im, (i,y_cord))
                #print(i, y_cord)
                i=(i+scaled_img_width)+padding
                j+=1

            new_im.save(final_output_file, "JPEG", quality=80, optimize=True, progressive=True)
            #new_im.show()

            # open the final combined file
            open_file(final_output_file)

            try_again_prompt = [
                inquirer.List('try',
                              message="Try again?",
                              choices=['Yes', 'No'],
                          ),
            ]

            answer = inquirer.prompt(try_again_prompt)
            if answer['try']=="No":
                print('bye!!')
                break

        except Exception as e:
            print(e)
