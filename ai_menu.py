from model import CGAN
from denoise import resize_and_denoise
from style_transfer import Stylizer
import cv2
from utils import words
from bert_transformers import BertQuery


class AIMenu:
    """
    Wrap the whole AI Menu pipeline in this class.
    """

    def __init__(self, result_path='./static/tmp'):
        checkpoint_dir = './ckpt'

        self.cgan = CGAN()
        self.bcq = BertQuery(model_dir='./ckpt/transformers')
        self.stylizer = Stylizer()

        self.cgan.reload(checkpoint_dir)
        self.result_path = result_path

    def get_topk_idx(self, description, topk=10):
        return self.bcq.query(description, topk=topk)

    def generate_character(self, topk_idx):
        img = self.cgan.generate_one_image(topk_idx)[0]
        img = (img * 0.5 + 0.5) * 255
        img = img.astype('uint8').squeeze()

        return img

    def style_transfer(self, style_image_path, content_img, output_size):
        return self.stylizer.transfer(style_image_path=style_image_path,
                                      content_image=content_img,
                                      output_size=output_size
                                      )

    def generate(self, description, style_img_file, denoise=True, result_size=(600, 600)):
        """Generate images according to dish name, and stylize the result according to style image.
        This function wrap the whole pipeline.

        Firstly, use Bert Client to generate a vector.
        Then use this vector to generate character using CGAN.
        Finally, denoise the result and stylize it.

        Return the original CGAN result, resized result, denoised result and stylized result.
        Notice that all the result are numpy array.
        And the value of the image is uint8 belong to [0, 255].

        :param denoise:
        :param description: dish name
        :param style_img_file: image for style transfer
        :param result_size: result size
        :return: img, resized_img, denoised_img, stylized_img
        """
        topk_idx = self.get_topk_idx(description)

        # do not use first 5 words because they could be similar when input dish names.
        # generate a character
        print('generating the character...')
        img = self.generate_character(topk_idx[5:])

        if denoise:
            # resize and denoise
            # denoising will change the style transfer effect to the background
            print('resizing and denoising the character...')
            resized_img, denoised_img = resize_and_denoise(img, result_size)
        else:
            img = img.astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, result_size, interpolation=cv2.INTER_AREA)
            dst = cv2.fastNlMeansDenoisingColored(img, None, 100, 100, 11, 27)
            resized_img = denoised_img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

        # style transfer
        print('stylizing the character...')
        stylized_img = self.style_transfer(style_image_path=style_img_file,
                                           content_img=denoised_img,
                                           output_size=1200)

        return img, resized_img, denoised_img, stylized_img, topk_idx[5:]
