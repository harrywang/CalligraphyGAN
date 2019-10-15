from model import CGAN
from denoise import *
import os
from bert_client import BertClientQuery
import datetime
import time


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

if __name__ == '__main__':
    bcq = BertClientQuery(words, topk=10)

    cgan = CGAN()
    cgan.reload('./ckpt')

    while True:
        query = input('Your query: ')
        topk_idx = bcq.query(query)

        time_now = str(int(time.time()))

        try:
            cgan.generate_one_image(topk_idx[5:], result_path='./result/%s.png' % time_now)
            resize_and_denoise('./result/%s.png' % time_now, (1200, 1200), './result/%s_convert.png' % time_now)
        except Exception as e:
            print(e)
