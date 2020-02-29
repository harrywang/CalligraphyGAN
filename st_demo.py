import streamlit as st
import numpy as np
from oil_painting import OilPaint
from ai_menu import AIMenu
from denoise import resize_and_denoise
import cv2
from utils import words, color_cluster, get_style_dict
import os


@st.cache(allow_output_mutation=True)
def init():
    # init menu images
    menu_images_path = './menu_image'
    menu_images = {}
    for item in os.listdir(menu_images_path):
        menu_images[item.split('.')[0]] = os.path.join(menu_images_path, item)

    # init styles
    styles = get_style_dict()

    # init AIMenu for image generating
    ai_menu = AIMenu(result_path='./static/tmp', topk=10)
    return menu_images, styles, ai_menu


@st.cache(allow_output_mutation=False)
def color_cluster_wrapper(image_path, topk=5):
    return color_cluster(image_path, topk)


def main():
    menu_images, styles, ai_menu = init()
    st.title('Food + Calligraphy + AI')
    st.header('Beautiful Enough to Feast the Eyes (秀色可餐)')

    st.sidebar.title('Configuration')

    st.sidebar.subheader('Dish Name')

    def convert_flag(x):
        if x:
            return 'Choose an existing dish'
        else:
            return 'Type my own dish name in Chinese'

    flag = st.sidebar.radio(label='', options=(True, False), format_func=convert_flag)
    target_color = None

    if flag:
        dish_name = st.sidebar.selectbox(label='Select a dish name', options=tuple(menu_images.keys()))
        st.image(menu_images[dish_name], width=300, caption=dish_name)
    else:
        dish_name = st.sidebar.text_input('Input your dish name', '鱼香肉丝')

    st.sidebar.subheader('Style')

    style_name = st.sidebar.radio(
        label="Choose a style for style transfer.",
        options=list(styles.keys()),
        key='style'
    )

    st.sidebar.image(styles[style_name], width=300, caption=style_name)
    st.sidebar.subheader('Threshold for Denoising')
    denoise_th = st.sidebar.slider(
        label='Pixel with value more than threshold will be convert to 255.',
        min_value=0, max_value=255, step=1, value=127)

    st.sidebar.subheader('Number of Color Clusters')
    number_color = st.sidebar.slider(
        label='Number of colors to paint.',
        min_value=3, max_value=7, step=1, value=5
    )

    st.sidebar.subheader('Number of Characters')
    number_characters = st.sidebar.slider(
        label='Number of characters to generate result.',
        min_value=3, max_value=10, step=1, value=5
    )

    if st.sidebar.button('Generate'):
        st.subheader('Result')

        color_cluster_image = st.empty()
        used_words = st.empty()
        denoised_image_ph = st.empty()
        stylized_image_ph = st.empty()
        oil_image_ph = st.empty()
        with st.spinner('Generating...'):
            # do not denoise the image for better performance
            if flag:
                color_image, target_color = color_cluster_wrapper(menu_images[dish_name], topk=number_color)
                color_cluster_image.image(color_image, width=300, caption='color cluster')
                target_color = np.array(target_color) / 255.

            topk_idx = ai_menu.get_topk_idx(dish_name, topk=number_characters * 2)[number_characters:]
            used_words.markdown('The model uses **%s** to generate the image.' % ','.join([words[idx] for idx in topk_idx]))

            img = ai_menu.generate_character(topk_idx)

            # denoise and resize
            _, denoised_img = resize_and_denoise(img, (300, 300), denoise_th)

            denoised_image_ph.image(denoised_img, width=300,
                                    caption='generated from %s' % ','.join([words[idx] for idx in topk_idx]))

            stylized_img = ai_menu.style_transfer(style_image_path=styles[style_name],
                                                  content_img=denoised_img,
                                                  output_size=300)

            stylized_image_ph.image(stylized_img, width=300, caption='Generated image stylized based on %s' % style_name)

            op = OilPaint(image=cv2.cvtColor(np.array(denoised_img), cv2.COLOR_RGB2BGR),
                          target_color=target_color)
            # smaller number for epoch and batch_size for better performance
            oil_img = op.paint(epoch=10, batch_size=32, result_dir=None) / 255.

            oil_image_ph.image(oil_img, width=300, caption='Generated image in oil painting style')
            st.balloons()


if __name__ == '__main__':
    main()
