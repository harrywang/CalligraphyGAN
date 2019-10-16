from flask import Flask, escape, request, flash
from flask_restful import Resource, Api, reqparse
from model import CGAN
from bert_client import BertClientQuery
from denoise import resize_and_denoise
from style_transfer import Stylizer
import time


# Initialization
app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class AIMenu(Resource):
    def __init__(self):
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

        checkpoint_dir = './ckpt'
        result_path = './result'

        self.cgan = CGAN()
        self.bcq = BertClientQuery(words=words, topk=10)
        self.stylizer = Stylizer()

        self.cgan.reload(checkpoint_dir)
        self.result_path = result_path

    def get(self):
        return {'message': 'you are sending a GET request to AIMenu'}

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('name')
        args = parser.parse_args()
        dish_name = args['name']    # get dish_name from parser

        if dish_name is None:
            return {'message': 'Error: dish name cannot be null'}

        time_now = str(int(time.time()))

        topk_idx = self.bcq.query(dish_name)

        # do not use first 5 words because they could be similar when input dish names.
        self.cgan.generate_one_image(topk_idx[5:], '%s/%s.png' % (self.result_path, time_now))
        resize_and_denoise(img_path='%s/%s.png' % (self.result_path, time_now),
                           result_size=(1200, 1200),
                           dst_path='%s/%s_convert.png' % (self.result_path, time_now))

        # change style_image_path to generate different style!
        # if something wrong with style_image_path, it will use default style image
        self.stylizer.transfer(style_image_path='./style_image/style01.jpg',
                               content_image_path='%s/%s_convert.png' % (self.result_path, time_now),
                               output_size=1200,
                               result_path='%s/%s_stylized.png' % (self.result_path, time_now)
                               )

        return {
            'url1': '%s/%s_convert.png' % (self.result_path, time_now),
            'url2': '%s/%s_stylized.png' % (self.result_path, time_now)
        }


api.add_resource(HelloWorld, '/')
api.add_resource(AIMenu, '/ai_menu')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
