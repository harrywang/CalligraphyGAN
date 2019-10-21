from model import CGAN
from bert_client import BertClientQuery
from denoise import resize_and_denoise
from style_transfer import Stylizer
import time


class AIMenu:
    def __init__(self, result_path='./static/tmp', topk=10):
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

        self.cgan = CGAN()
        self.bcq = BertClientQuery(words=words, topk=topk)
        self.stylizer = Stylizer()

        self.cgan.reload(checkpoint_dir)
        self.result_path = result_path

    def generate(self, description):
        time_now = str(int(time.time()))

        topk_idx = self.bcq.query(description)

        # do not use first 5 words because they could be similar when input dish names.
        self.cgan.generate_one_image(topk_idx[5:], '%s/%s.png' % (self.result_path, time_now))
        resize_and_denoise(img_path='%s/%s.png' % (self.result_path, time_now),
                           result_size=(1200, 1200),
                           dst_path='%s/%s_convert.png' % (self.result_path, time_now))

        # change style_image_path to generate different style!
        # if something wrong with style_image_path, it will use default style image
        self.stylizer.transfer(style_image_path='./style_image/dekooning.jpg',
                               content_image_path='%s/%s_convert.png' % (self.result_path, time_now),
                               output_size=1200,
                               result_path='%s/%s_stylized.png' % (self.result_path, time_now)
                               )

        return {
            'url1': '%s/%s_convert.png' % (self.result_path, time_now),
            'url2': '%s/%s_stylized.png' % (self.result_path, time_now)
        }
