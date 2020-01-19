from flask import Flask, escape, request, flash
from flask_restful import Resource, Api, reqparse
from model import CGAN
from bert_client import BertClientQuery
from denoise import resize_and_denoise
from style_transfer import Stylizer
import time
from PIL import Image
import numpy as np
from ai_menu import AIMenu

# Initialization
app = Flask(__name__)
api = Api(app)


class AIMenuAPI(Resource):
    def __init__(self, checkpoint_dir='./ckpt', result_path='./result'):
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

        self.cgan = CGAN()
        self.bcq = BertClientQuery(words=words, topk=10)
        self.stylizer = Stylizer()

        self.cgan.reload(checkpoint_dir)
        self.result_path = result_path
        self.style_images = {
            'dekooning': './style_image/dekooning.jpg',
            'picasso': './style_image/picasso.jpg',
            'pollock': './style_image/pollock.jpg',
            'rousseau': './style_image/rousseau.jpg',
            'rothko': './style_image/rothko.jpg'
        }

        self.menu = AIMenu(result_path=self.result_path, topk=10)

    def get(self):
        return {'message': 'you are sending a GET request to AIMenu'}

    # require 2 args, `name` for dish_name and `style` for style image.
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        parser.add_argument('style')

        args = parser.parse_args()
        dish_name = args['name']  # get dish_name from parser
        style_image = self.style_images[args['style'].lower()]

        if dish_name is None:
            return {'message': 'Error: dish name cannot be null'}

        dst_path = '%s/%s' % (self.result_path, str(int(time.time())))

        img, resized_img, denoised_img, stylized_img = self.menu.generate(dish_name, style_image)

        stylized_img = Image.fromarray(stylized_img.astype(np.uint8))
        denoised_img = Image.fromarray(denoised_img.astype(np.uint8))
        stylized_img.save('%s_stylized.png' % dst_path)
        denoised_img.save('%s_convert.png' % dst_path)

        # TODO: url should be changed according to configuration of server
        return {
            'url_convert': '%s_convert.png' % dst_path,
            'url_stylized': '%s_stylized.png' % dst_path
        }


api.add_resource(AIMenuAPI, '/ai_menu')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
