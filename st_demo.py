import streamlit as st
import time
import os
from PIL import Image
import numpy as np
import time
from oil_painting import OilPaint
from ai_menu import AIMenu


styles = {
    'Dekooning': './style_image/dekooning.jpg',
    'Picasso': './style_image/picasso.jpg',
    'Pollock': './style_image/pollock.jpg',
    'Rousseau': './style_image/rousseau.jpg'
}


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
        denoised_image = st.empty()
        stylized_image = st.empty()
        oil_image = st.empty()
        with st.spinner('Generating...') as info:
            img, resized_img, denoised_img, stylized_img = ai_menu.generate(description=dish_name,
                                                                            style_img_file=styles[style_name])
            denoised_image.image(resized_img, width=300, caption='generated image')
            stylized_image.image(stylized_img, width=300, caption='stylized image')

            # TODO: no need to save images, so how OilPaint get resized image.
            # op = OilPaint(file_name=resized_file)
            # oil_img = op.paint(epoch=30, batch_size=64, result_dir=None)
            #
            # oil_image.image(oil_img, width=300, caption='oiled image')
            st.balloons()


if __name__ == '__main__':
    main()
