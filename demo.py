# -*- coding: utf-8 -*-
import os
import time
import platform
import subprocess
import inquirer
import numpy as np
import glob
from PIL import Image
from math import ceil, floor
from ai_menu import AIMenu
from oil_painting import OilPaint
from utils import words


def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


if __name__ == '__main__':
    print('initializing ......')
    menu = AIMenu(result_path='./result', topk=10)

    while True:
        query = input("Enter dish name - [default:红烧肉卤鸡蛋]:")
        if len(query) == 0:
            query = "红烧肉卤鸡蛋"

        # topk_idx = bcq.query(query)
        # print(topk_idx)

        time_now = str(int(time.time()))
        charactor_file = './result/%s.png' % time_now
        resized_file = './result/%s_convert.png' % time_now
        styled_file = './result/%s_stylized.png' % time_now
        final_output_file = './result/%s_final.jpg' % time_now
        oil_file = './result/%s_oil.png' % time_now
        oil_dir = './result/%s_oil' % time_now

        try:
            # generate a character

            # TODO: inquirer is not supported on Windows
            styles = [
                inquirer.List('style',
                              message="Choose a style:",
                              choices=['Picasso', 'Pollock', 'Rousseau', 'Rothko', 'deKooning'],
                              ),
            ]

            chosen_style = inquirer.prompt(styles)
            style_img_file = './style_image/%s.jpg' % chosen_style['style'].lower()

            img, resized_img, denoised_img, stylized_img, _ = menu.generate(query, style_img_file=style_img_file)

            Image.fromarray(img).convert('RGB').save(charactor_file)
            Image.fromarray(denoised_img.astype(np.uint8)).save(resized_file)
            Image.fromarray(stylized_img.astype(np.uint8)).save(styled_file)

            print('start oil painting...')
            if not os.path.exists(oil_dir):
                os.makedirs(oil_dir)
            op = OilPaint(file_name=resized_file)
            oil_img = op.paint(epoch=30, batch_size=64, result_dir=oil_dir)

            Image.fromarray(oil_img.astype('uint8')).save(oil_file)

            # combine the output images into one
            print('Finishing up...')
            frame_width = 1920
            images_per_row = 5
            padding = 2

            images = glob.glob('./result/%s*.png' % time_now)
            images.append(style_img_file)
            images.sort()

            img_width, img_height = Image.open(images[0]).size
            sf = (frame_width - (images_per_row - 1) * padding) / (images_per_row * img_width)  # scaling factor
            scaled_img_width = ceil(img_width * sf)  # s
            scaled_img_height = ceil(img_height * sf)

            number_of_rows = ceil(len(images) / images_per_row)
            frame_height = ceil(sf * img_height * number_of_rows)

            new_im = Image.new('RGB', (frame_width, frame_height))

            i, j = 0, 0
            for num, im in enumerate(images):
                if num % images_per_row == 0:
                    i = 0
                im = Image.open(im)
                # Here I resize my opened image, so it is no bigger than 100,100
                im.thumbnail((scaled_img_width, scaled_img_height))
                # Iterate through a 5 by 5 grid with 100 spacing, to place my image
                y_cord = (j // images_per_row) * scaled_img_height
                new_im.paste(im, (i, y_cord))
                # print(i, y_cord)
                i = (i + scaled_img_width) + padding
                j += 1

            new_im.save(final_output_file, "JPEG", quality=80, optimize=True, progressive=True)

            # TODO: inquirer is not supported on Windows
            try_again_prompt = [
                inquirer.List('try',
                              message="Try again?",
                              choices=['Yes', 'No'],
                              ),
            ]

            answer = inquirer.prompt(try_again_prompt)
            if answer['try'] == "No":
                print('bye!!')
                break

        except Exception as e:
            print(e)
