import streamlit as st
import numpy as np
from models.oil_painting import OilPaint
from models.calligraphyGAN_dataset import _get_embedding
from ai_menu import AIMenu
from utils.denoise import resize_and_denoise
import cv2
from utils.aestheic_filter import WhiteSpaceFilter
import os
from sklearn.cluster import KMeans


def get_style_dict():
    style_image_path = './style_image'
    result = {}
    for item in os.listdir(style_image_path):
        result[item.split('.')[0]] = os.path.join(style_image_path, item)

    return result


# function to get topk color in image
def color_cluster(image_path, topk=5):
    """
    get top-k colors in an image
    :param topk:
    :param image_path:
    :return:
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # cluster the pixel intensities
    clt = KMeans(n_clusters=topk)
    clt.fit(image)

    # below is the code to visualize the results
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)
    def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        # return the histogram
        return hist

    def plot_colors(hist, centroids):
        # initialize the bar chart representing the relative frequency
        # of each of the colors
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        # loop over the percentage of each cluster and the color of
        # each cluster
        for (percent, color) in zip(hist, centroids):
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX

        # return the bar chart
        return bar

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    return bar, clt.cluster_centers_
    # # show our color bart
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(bar)
    # plt.show()


@st.cache(allow_output_mutation=True)
def init():
    # init menu images
    menu_images = {
        '白灼生菜 Boiled Lettuce': './menu_image/Boiled Lettuce.png',
        '法式意大利黑醋带鱼 Italian Black Vinegar Hairtail': './menu_image/Italian Black Vinegar Hairtail.png',
        '家烧三门青膏蟹 Homemade Style Green Crabs': './menu_image/Homemade Style Green Crabs.png',
        '椒香美味裙 Roasted Scollop Apron': './menu_image/Roasted Scollop Apron.png',
        '松仁丝瓜尖 Luffa Seedlings with Pine Nuts': './menu_image/Luffa Seedlings with Pine Nuts.png',
    }

    # init styles
    styles = get_style_dict()

    # init AIMenu for image generating
    ai_menu = AIMenu(result_path='./static/tmp', bert_model_path='./ckpt/transformers')

    words = _get_embedding('./data/label_character.csv')
    # _get_embedding return the dict like { ..., character: embedding, ... }
    # convert to a list like [ ..., character, ... ]
    words = list(words.keys())

    return menu_images, styles, ai_menu, words


@st.cache(allow_output_mutation=False)
def color_cluster_wrapper(image_path, topk=5):
    return color_cluster(image_path, topk)


def main():
    # create some folders
    if not os.path.exists('./static/tmp'):
        os.makedirs('./static/tmp')

    menu_images, styles, ai_menu,words = init()
    st.title('Abstract Art via CalligraphyGAN')
    st.sidebar.title('Configuration')
    st.sidebar.subheader('Dish Name')

    # convert flag (True or False) to corresponding instruction
    def convert_flag_to_instruction(x):
        if x:
            return 'Choose an existing dish'
        else:
            return 'Type my own dish name in Chinese'

    use_existing_name = st.sidebar.radio(label='', options=(True, False), format_func=convert_flag_to_instruction)
    target_color = None

    if use_existing_name:
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

    # I set threshold for denoising same as `white_threshold` in White Space Ratio settings.
    # st.sidebar.subheader('Threshold for Denoising')
    # denoise_th = st.sidebar.slider(
    #     label='Pixel with value more than threshold will be convert to 255.',
    #     min_value=0, max_value=255, step=1, value=127)

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

    st.sidebar.subheader('Threshold of White Space Ratio')
    white_space_lower = st.sidebar.slider(
        label='Lower bound of white space ratio.',
        min_value=0., max_value=1., step=0.1, value=0.3
    )
    white_space_upper = st.sidebar.slider(
        label='Upper bound of white space ratio.',
        min_value=0., max_value=1., step=0.1, value=0.6
    )
    white_threshold = st.sidebar.slider(
        label='Pixel will be considered as white when its value is larger than this number.',
        min_value=0, max_value=255, step=1, value=127
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
            if use_existing_name:
                color_image, target_color = color_cluster_wrapper(menu_images[dish_name], topk=number_color)
                color_cluster_image.image(color_image, width=300, caption='color cluster')
                target_color = np.array(target_color) / 255.

            topk_idx = ai_menu.get_topk_idx(dish_name, topk=number_characters * 2)[number_characters:]
            used_words.markdown(
                'The model uses **%s** to generate the image.' % ','.join([words[idx] for idx in topk_idx]))

            filters = [
                WhiteSpaceFilter(t_min=white_space_lower, t_max=white_space_upper, white_threshold=white_threshold)
            ]
            # TODO: How to show result when all generated result are filtered
            img = ai_menu.generate_character_with_filter(topk_idx=topk_idx, number=100, filters=filters, topk=1)[0]

            # denoise and resize
            _, denoised_img = resize_and_denoise(img, (300, 300), white_threshold)
            denoised_img = (255. - denoised_img).astype(np.uint8)

            denoised_image_ph.image(denoised_img, width=300,
                                    caption='generated from %s' % ','.join([words[idx] for idx in topk_idx]))

            stylized_img = ai_menu.style_transfer(style_image_path=styles[style_name],
                                                  content_img=denoised_img,
                                                  output_size=300)

            stylized_image_ph.image(stylized_img, width=300,
                                    caption='Generated image stylized based on %s' % style_name)

            op = OilPaint(image=cv2.cvtColor(np.array(denoised_img), cv2.COLOR_RGB2BGR),
                          target_color=target_color, brush_dir='./data/brushes')

            # smaller number for epoch and batch_size for better performance
            oil_img = op.paint(epoch=10, batch_size=32, result_dir=None) / 255.

            oil_image_ph.image(oil_img, width=300, caption='Generated image in oil painting style')
            st.balloons()


if __name__ == '__main__':
    main()
