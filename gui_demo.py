from flask import Flask, escape, request, flash, render_template, url_for
from ai_menu import AIMenu
import os
from PIL import Image
import numpy as np
import time
from oil_painting import OilPaint
import cv2
from utils import styles


if not os.path.exists('./static/tmp'):
    os.makedirs('./static/tmp')

app = Flask(__name__)
ai_menu = AIMenu(result_path='./static/tmp', topk=10)


@app.route('/', methods=['GET', 'POST'])
def generate():
    if request.method == 'GET':
        return render_template('cgan.html', generated=False)
    elif request.method == 'POST':
        desc = request.form['description']
        style = request.form['style']
        if style not in styles.keys():
            # if not choose a style or something wrong with it, pick one randomly
            import random
            style = random.choice(list(styles.keys()))

        img, resized_img, denoised_img, stylized_img, _ = ai_menu.generate(desc, style_img_file=styles[style])

        time_now = str(int(time.time()))
        resized_file = './static/tmp/%s_convert.png' % time_now
        styled_file = './static/tmp/%s_stylized.png' % time_now
        oil_file = './static/tmp/%s_oil.png' % time_now
        oil_dir = './static/tmp/%s_oil' % time_now

        Image.fromarray(denoised_img.astype(np.uint8)).save(resized_file)
        Image.fromarray(stylized_img.astype(np.uint8)).save(styled_file)

        op = OilPaint(cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR))
        if not os.path.exists(oil_dir):
            os.makedirs(oil_dir)
        oil_img = op.paint(epoch=30, batch_size=64, result_dir=oil_dir)
        Image.fromarray(oil_img.astype('uint8')).save(oil_file)

        return render_template('cgan.html', generated=True,
                               url_1=resized_file,
                               url_2=styled_file,
                               url_3=oil_file,
                               desc=desc,
                               style=style)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
