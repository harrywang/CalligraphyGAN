from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from skimage.transform import resize
import cv2
from sklearn.cluster import KMeans
import utils
import cv2

words_bak = ['且', '世', '东', '九', '亭', '今', '从', '令', '作', '使',
             '侯', '元', '光', '利', '印', '去', '受', '右', '司', '合',
             '名', '周', '命', '和', '唯', '堂', '士', '多', '夜', '奉',
             '女', '好', '始', '字', '孝', '守', '宗', '官', '定', '宜',
             '室', '家', '寒', '左', '常', '建', '徐', '御', '必', '思',
             '意', '我', '敬', '新', '易', '春', '更', '朝', '李', '来',
             '林', '正', '武', '氏', '永', '流', '海', '深', '清', '游',
             '父', '物', '玉', '用', '申', '白', '皇', '益', '福', '秋',
             '立', '章', '老', '臣', '良', '莫', '虎', '衣', '西', '起',
             '足', '身', '通', '遂', '重', '陵', '雨', '高', '黄', '鼎']

words = ['一', '丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丑', '且', '世', '丘', '丙', '业', '东', '丝', '丞', '两', '严', '中',
         '丰', '临', '丹', '为', '主', '丽', '举', '乃', '久', '么', '义', '之', '乍', '乎', '乐', '乘', '乙', '九', '也', '乡', '书', '乱',
         '了', '予', '争', '事', '二', '于', '云', '五', '井', '亡', '交', '亦', '产', '享', '京', '亭', '亲', '人', '今', '介', '仍', '从',
         '他', '仙', '代', '令', '以', '仲', '价', '任', '伊', '伏', '休', '众', '伙', '会', '传', '伪', '伯', '似', '但', '体', '何', '作',
         '佣', '佳', '使', '侯', '侵', '俊', '修', '假', '健', '僕', '儿', '元', '兄', '兆', '先', '光', '克', '免', '党', '入', '全', '八',
         '公', '六', '兰', '关', '兴', '兵', '其', '具', '兹', '兼', '内', '册', '再', '冑', '写', '农', '冠', '冢', '冬', '况', '冷', '准',
         '凌', '几', '凡', '凤', '出', '刃', '分', '切', '划', '列', '刘', '则', '初', '别', '到', '制', '刻', '前', '剑', '副', '力', '功',
         '加', '务', '动', '助', '劳', '势', '勉', '勤', '勿', '包', '化', '北', '区', '医', '十', '千', '升', '午', '半', '华', '卒', '单',
         '南', '博', '占', '卢', '卯', '危', '即', '却', '卷', '压', '厌', '原', '厥', '去', '县', '参', '又', '及', '友', '双', '反', '发',
         '叔', '取', '受', '变', '古', '句', '只', '可', '台', '史', '右', '叶', '号', '司', '吁', '各', '合', '吉', '吊', '同', '名', '后',
         '向', '君', '吟', '听', '启', '吴', '吹', '吾', '告', '周', '命', '咏', '咨', '咸', '哀', '哉', '响', '哥', '唐', '唯', '商', '善',
         '喜', '嗓', '嘉', '器', '四', '回', '因', '园', '固', '国', '图', '圆', '土', '圣', '在', '地', '坏', '坐', '坡', '垂', '城', '堂',
         '堪', '塞', '墅', '墨', '壁', '壑', '士', '壮', '声', '处', '备', '夏', '夕', '外', '多', '夜', '大', '天', '太', '失', '头', '夷',
         '夸', '夹', '奇', '奉', '契', '奔', '女', '奶', '好', '如', '妇', '妙', '始', '姓', '姜', '威', '子', '孔', '字', '存', '孙', '孝',
         '孤', '学', '宁', '宇', '守', '安', '宋', '宗', '官', '宙', '定', '实', '客', '宣', '室', '宫', '家', '容', '宾', '宿', '寂', '寄',
         '密', '寒', '寥', '对', '寺', '寻', '寿', '封', '将', '尉', '尊', '小', '少', '尔', '尘', '尚', '尝', '尤', '就', '尸', '尹', '尽',
         '局', '居', '展', '属', '履', '山', '岁', '岂', '岱', '岳', '岸', '峰', '峻', '崇', '崑', '崖', '嵩', '巖', '川', '州', '工', '左',
         '巫', '差', '己', '巳', '巾', '市', '布', '帆', '师', '希', '帘', '帝', '带', '席', '常', '幕', '干', '平', '年', '并', '幸', '幽',
         '庄', '庆', '序', '应', '庙', '府', '废', '度', '座', '庶', '康', '庸', '廉', '延', '建', '开', '异', '弃', '弈', '弊', '式', '引',
         '弗', '弘', '弟', '张', '弥', '弯', '弱', '归', '当', '形', '彦', '彩', '彭', '影', '往', '徒', '得', '御', '復', '微', '德', '心',
         '必', '忍', '志', '忘', '忠', '忧', '快', '念', '忽', '怀', '怜', '思', '性', '总', '恐', '恭', '息', '恶', '悠', '患', '情', '惊',
         '惜', '惟', '惠', '意', '感', '愿', '慰', '懿', '戊', '戎', '戏', '成', '我', '戒', '或', '战', '所', '手', '才', '扎', '扑', '打',
         '执', '扬', '承', '投', '折', '抟', '招', '拜', '拥', '拼', '持', '挂', '指', '按', '捧', '据', '捻', '授', '推', '掺', '摄', '摇',
         '支', '收', '放', '政', '故', '救', '教', '敝', '敢', '散', '敦', '敬', '数', '整', '文', '斋', '斗', '斛', '斜', '斤', '断', '斯',
         '新', '方', '旅', '无', '既', '日', '旦', '旧', '旨', '早', '时', '昌', '明', '易', '昔', '星', '映', '春', '昨', '是', '晋', '晏',
         '晓', '晚', '普', '景', '曰', '曲', '更', '曹', '曼', '曾', '最', '月', '有', '朋', '服', '望', '朝', '木', '未', '末', '本', '朴',
         '李', '村', '杜', '条', '来', '杰', '松', '极', '林', '果', '枝', '柔', '柳', '树', '栖', '根', '格', '桃', '桑', '桥', '梁', '梅',
         '梦', '楚', '楼', '檐', '次', '欲', '歌', '正', '此', '武', '歷', '死', '殊', '殿', '每', '比', '毫', '氏', '民', '气', '水', '永',
         '氾', '求', '汉', '江', '池', '污', '沈', '沓', '沖', '沙', '河', '泉', '法', '泛', '波', '洒', '洞', '洼', '流', '济', '浓', '海',
         '涂', '涌', '淡', '深', '清', '渊', '渔', '游', '湖', '湾', '溪', '溼', '满', '漫', '火', '灵', '点', '烟', '热', '焉', '然', '照',
         '熟', '燕', '爱', '片', '物', '犹', '独', '猪', '献', '玄', '率', '玉', '王', '珍', '班', '理', '璇', '瓜', '甘', '甚', '生', '用',
         '甫', '田', '由', '画', '畅', '界', '留', '病', '痒', '痴', '登', '白', '百', '的', '皆', '皇', '益', '盖', '盘', '盛', '盥', '目',
         '盲', '直', '相', '看', '真', '眼', '着', '睢', '矣', '知', '石', '砖', '破', '碧', '示', '礼', '神', '禄', '禅', '福', '离', '禽',
         '秀', '秉', '秋', '种', '秦', '积', '称', '稽', '穆', '穷', '空', '立', '竟', '章', '端', '竹', '笋', '笑', '笔', '第', '等', '筑',
         '米', '类', '精', '系', '素', '紫', '约', '纷', '纸', '细', '终', '经', '给', '绝', '继', '绿', '置', '美', '群', '老', '者', '而',
         '耳', '联', '聚', '肃', '胜', '能', '脏', '脩', '腊', '腌', '腔', '膺', '臣', '自', '至', '致', '臺', '舍', '舞', '舟', '船', '良',
         '色', '艺', '节', '花', '芳', '苍', '苏', '若', '苦', '英', '茂', '范', '茶', '草', '荐', '荒', '荣', '药', '莫', '获', '萧', '落',
         '蒙', '薄', '藏', '虎', '虚', '虞', '虫', '虽', '蛇', '蜀', '蜡', '行', '衣', '衰', '被', '襄', '西', '覆', '见', '观', '觉', '解',
         '言', '謚', '警', '议', '记', '许', '论', '访', '识', '诊', '试', '诗', '诚', '语', '请', '诸', '读', '谁', '谓', '谢', '谷', '象',
         '豫', '财', '贤', '贵', '贺', '资', '赠', '走', '赵', '起', '趣', '足', '跖', '路', '跳', '踊', '身', '转', '轻', '载', '辈', '辛',
         '辞', '辨', '辰', '边', '迁', '过', '近', '还', '远', '述', '迹', '适', '通', '造', '遂', '道', '遗', '邑', '邮', '郁', '郎', '郑',
         '郡', '部', '都', '酒', '醉', '醒', '采', '里', '重', '野', '金', '钱', '铁', '铜', '铭', '锈', '长', '门', '问', '闲', '间', '闹',
         '闻', '阁', '阳', '阴', '际', '陈', '陋', '降', '除', '陵', '陶', '隆', '随', '隐', '隶', '难', '雄', '雅', '集', '雨', '雪', '雷',
         '雾', '霁', '霉', '霜', '青', '静', '非', '面', '音', '须', '顾', '顿', '领', '颇', '频', '颖', '题', '颜', '风', '飞', '食', '餐',
         '饥', '饭', '饮', '馀', '馆', '首', '香', '马', '驰', '骑', '骨', '高', '髮', '鬼', '魂', '魏', '鱼', '鲁', '鸟', '鸣', '鸿', '鹅',
         '鹤', '鹿', '黄', '黑', '鼓', '鼠', '齐', '齿', '龙', '龟']

# styles = {
#     'Dekooning': './style_image/dekooning.jpg',
#     'Picasso': './style_image/picasso.jpg',
#     'Pollock': './style_image/pollock.jpg',
#     'Rousseau': './style_image/rousseau.jpg',
#     'Rothko': './style_image/rothko.jpg'
# }

style_image_path = './style_image'


def get_style_dict():
    result = {}
    for item in os.listdir(style_image_path):
        result[item.split('.')[0]] = os.path.join(style_image_path, item)

    return result


# rewrite cv2 imread to read files with Chinese characters
def cv_read_img_BGR(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# rewrite cv2 imwrite to write files with Chinese characters
def cv_write_img_BGR(img, result_path):
    cv2.imencode('.png', img)[1].tofile(result_path)


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

    # below is the code to visualize the resultsv
    # plt.figure()
    # plt.axis("off")
    # plt.imshow(image)
    #
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


def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def save_one_sample_image(sample_images, result_path, is_square=True):
    if len(sample_images.shape) == 2:
        size = int(np.sqrt(sample_images.shape[1]))
    elif len(sample_images.shape) > 2:
        size = sample_images.shape[1]
        channel = sample_images.shape[3]
    else:
        raise ValueError('Not valid a shape of sample_images')

    if not is_square:
        print_images = sample_images[:1, ...]
        print_images = print_images.reshape([1, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, 1 * size, channel])
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(1, 1))
        plt.imshow(print_images * 0.5 + 0.5)  # , cmap='gray')
        plt.axis('off')

    else:
        num_columns = int(np.sqrt(1))
        max_print_size = int(num_columns ** 2)
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        print_images = [print_images[:, i * size * num_columns:(i + 1) * size * num_columns] for i in
                        range(num_columns)]
        print_images = np.concatenate(tuple(print_images), axis=0)
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(num_columns, num_columns))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.imshow(print_images * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    plt.savefig(result_path, dpi=300)
    plt.close()


def print_or_save_sample_images(sample_images, max_print_size=25,
                                is_square=False, is_save=False, epoch=None,
                                checkpoint_dir='./train'):
    available_print_size = list(range(1, 101))
    assert max_print_size in available_print_size
    if len(sample_images.shape) == 2:
        size = int(np.sqrt(sample_images.shape[1]))
    elif len(sample_images.shape) > 2:
        size = sample_images.shape[1]
        channel = sample_images.shape[3]
    else:
        ValueError('Not valid a shape of sample_images')

    if not is_square:
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(max_print_size, 1))
        plt.imshow(print_images * 0.5 + 0.5)  # , cmap='gray')
        plt.axis('off')

    else:
        num_columns = int(np.sqrt(max_print_size))
        max_print_size = int(num_columns ** 2)
        print_images = sample_images[:max_print_size, ...]
        print_images = print_images.reshape([max_print_size, size, size, channel])
        print_images = print_images.swapaxes(0, 1)
        print_images = print_images.reshape([size, max_print_size * size, channel])
        print_images = [print_images[:, i * size * num_columns:(i + 1) * size * num_columns] for i in
                        range(num_columns)]
        print_images = np.concatenate(tuple(print_images), axis=0)
        if channel == 1:
            print_images = np.squeeze(print_images, axis=-1)

        fig = plt.figure(figsize=(num_columns, num_columns))
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.imshow(print_images * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    if is_save and epoch is not None:
        filepath = os.path.join(checkpoint_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(filepath, dpi=300)
    else:
        plt.show()


# TODO: rewrite this function
def generate_gif(vector, result_dir):
    with imageio.get_writer(os.path.join(result_dir, 'result.gif'), mode='I') as writer:
        for j in range(0, 40):
            image = imageio.imread(os.path.join(result_dir, '%d.png' % j))
            writer.append_data(image)

        for j in range(0, 10):
            image = imageio.imread(os.path.join(result_dir, '39.png'))
            writer.append_data(image)

        for j in range(39, 0, -1):
            image = imageio.imread(os.path.join(result_dir, '%d.png' % j))
            writer.append_data(image)
