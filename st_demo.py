import streamlit as st
import numpy as np
from oil_painting import OilPaint
from ai_menu import AIMenu
import cv2
from utils import words, styles


@st.cache(allow_output_mutation=True)
def init():
    ai_menu = AIMenu(result_path='./static/tmp', topk=10)
    return ai_menu


def main():
    ai_menu = init()
    st.title('AI Receipt Art')
    dish_name = st.text_input('Dish Name', '鱼香肉丝')

    style_name = st.radio(
        label="Choose a style for style transfer.",
        options=list(styles.keys()),
        key='style'
    )

    st.image(styles[style_name], width=300, caption='style image')

    if st.button('Generate'):
        used_words = st.empty()
        denoised_image = st.empty()
        stylized_image = st.empty()
        oil_image = st.empty()
        with st.spinner('Generating...'):
            # do not denoise the image for better performance
            img, resized_img, denoised_img, stylized_img, topk_idx = ai_menu.generate(description=dish_name,
                                                                                      style_img_file=styles[style_name],
                                                                                      denoise=False,
                                                                                      result_size=(300, 300))
            used_words.markdown('The model uses **%s** to generate image.' % ','.join([words[idx] for idx in topk_idx]))
            denoised_image.image(resized_img, width=300, caption='generated image')
            stylized_image.image(stylized_img, width=300, caption='stylized image')

            op = OilPaint(cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR))
            # smaller number for epoch and batch_size for better performance
            oil_img = op.paint(epoch=10, batch_size=32, result_dir=None) / 255.

            oil_image.image(oil_img, width=300, caption='oiled image')
            st.balloons()


if __name__ == '__main__':
    main()
