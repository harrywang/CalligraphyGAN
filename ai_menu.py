from model import CGAN
from bert_client import BertClientQuery
from denoise import resize_and_denoise
from style_transfer import Stylizer


class AIMenu:
    """
    Wrap the whole AI Menu pipeline in this class.
    """
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

    def generate(self, description, style_img_file, result_size=(600, 600)):
        """Generate images according to dish name, and stylize the result according to style image.

        Firstly, use Bert Client to generate a vector.
        Then use this vector to generate character using CGAN.
        Finally, denoise the result and stylize it.

        Return the original CGAN result, resized result, denoised result and stylized result.
        Notice that all the result are numpy array.
        And the value of the image is uint8 belong to [0, 255].

        :param description: dish name
        :param style_img_file: image for style transfer
        :param result_size: result size
        :return: img, resized_img, denoised_img, stylized_img
        """
        topk_idx = self.bcq.query(description)

        # do not use first 5 words because they could be similar when input dish names.
        # generate a character
        print('generating the character...')
        img = self.cgan.generate_one_image(topk_idx[5:])[0]
        img = (img * 0.5 + 0.5) * 255
        img = img.astype('uint8').squeeze()

        # resize and denoise
        # denoising will change the style transfer effect to the background
        print('resizing and denoising the character...')
        resized_img, denoised_img = resize_and_denoise(img, result_size)

        # style transfer
        print('stylizing the character...')
        stylized_img = self.stylizer.transfer(style_image_path=style_img_file,
                                              content_image=denoised_img,
                                              output_size=1200
                                              )

        return img, resized_img, denoised_img, stylized_img
