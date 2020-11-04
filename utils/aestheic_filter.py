"""
This file implemented Aesthetic Filters.

AestheticFilter is the base class and it has 3 methods, _get_score, _score_2_index, and get_result.

Function get_result implemented in AestheticFilter should not be overridden.
In `get_result`, we firstly use _get_score to calculate scores of every image and then use _score_2_index to sort them
according to the need. Finally, return the top-k result. This is the general idea of Aesthetic Filters.

But how to get scores and sort them are different according to different purposes. So, subclasses can do different things
by override `_ger_score` and `_score_2_index`.

ThresholdFilter handle a special condition in which we want to filter results with threshold, which means we just want
to get results between the upper bound and lower bound.
In this implementation, I use a value's minimum distance to boundaries to sort them. The distance is calculated as
follows:
    d = min( value - lower-bound, upper-bound - value). So we just need to return positive distances.

WhiteSpaceFilter is a special case of ThresholdFilter.
"""
import numpy as np
from sklearn.cluster import DBSCAN


class AestheticFilter(object):
    def __init__(self):
        pass

    def _get_score(self, img):
        """
        :param img:
        :return: a list of scores
        """
        return 0.

    def _score_2_index(self, scores):
        """
        sort scores (max to min)
        :param scores:
        :return:
        """
        index = np.argsort(scores)
        index = np.flip(index, axis=0)
        return index

    def get_result(self, input_image, topk):
        """

        :param input_image:
        :param topk:
        :return: filtered images list
        """
        if len(input_image) < topk:
            print('Filter Warning: You have %d results but you want to get top%d.' % (len(input_image), topk))

        result = []
        scores = []
        for img in input_image:
            scores.append(self._get_score(img))
        index = self._score_2_index(scores)

        if len(index) < topk:
            print('Filter Warning: Only %d results left, less than value of topk: %d.' % (len(index), topk))
        else:
            index = index[:topk]

        for idx in index:
            result.append(input_image[idx])

        return result


class ThresholdFilter(AestheticFilter):
    def __init__(self, t_min, t_max):
        super(AestheticFilter, self).__init__()
        assert (t_min is not None or t_max is not None), 'Filter Error: thresholds cannot both be None.'

        self.t_min = t_min
        self.t_max = t_max

    def _score_2_index(self, scores):
        # calculate the min distance to threshold
        # positive means between [t_min, t_max]
        # negative means beyond the region
        # and the bigger distance, the closer to the middle of region
        for i, score in enumerate(scores):
            scores[i] = min(score - self.t_min if self.t_min is not None else float('inf'),
                            self.t_max - score if self.t_max is not None else float('inf'))

        index = np.argsort(scores)
        result = []
        for idx in index:
            if scores[idx] >= 0:
                result.append(idx)

        if len(result) == 0:
            print('Filter Warning: No score meet the threshold.')
            return index[0:1]
        return result


class WhiteSpaceFilter(ThresholdFilter):
    def __init__(self, t_min, t_max, white_threshold):
        super(WhiteSpaceFilter, self).__init__(t_min, t_max)
        self.white_threshold = white_threshold

    def _get_score(self, img):
        width, height = img.shape[0], img.shape[1]
        white_space_cnt = 0.
        for i in range(width):
            for j in range(height):
                if img[i][j] >= self.white_threshold:
                    white_space_cnt += 1
        return white_space_cnt / (width * height)


class RuleOfThirdFilter(AestheticFilter):
    def __init__(self):
        super(RuleOfThirdFilter, self).__init__()

    def _get_score(self, img):
        width, height = img.shape[1], img.shape[0]
        # calculate 4 third points and 4 third lines
        third_points = [
            (width / 3., height / 3.),
            (width / 3., height * 2. / 3.),
            (width * 2. / 3., height / 3.),
            (width * 2. / 3, height * 2. / 3.)
        ]
        third_lines_width = [
            width / 3.,
            width * 2. / 3.
        ]
        third_lines_height = [
            height / 3.,
            height * 2 / 3.
        ]

        # cluster the black points in image
        points = []
        for i in range(height):
            for j in range(width):
                if img[i][j] < 1.:
                    points.append((i, j))

        clt = DBSCAN(eps=5, min_samples=20).fit(points)

        # find the biggest component
        label_dict = {}
        max_index = 0
        for item in clt.labels_:
            if item in label_dict.keys():
                label_dict[item] += 1
                if label_dict[item] > label_dict[max_index]:
                    max_index = item
            else:
                label_dict[item] = 0

        # find the center of biggest component
        biggest_label = sorted(label_dict.items(), key=lambda d: d[1])[-1][0]
        biggest_component = []
        for idx, item in enumerate(clt.labels_):
            if item == biggest_label:
                biggest_component.append(points[idx])

        biggest_component = np.array(biggest_component)
        center = np.mean(biggest_component, axis=0)

        # calculate the min distance between center and four third points
        min_distance = float('inf')
        for point in third_points:
            distance = np.linalg.norm(center - point)
            min_distance = distance if distance < min_distance else min_distance

        # TODO: 中轴线到三等分线的距离
        # find the center axis of biggest component
        # calculate the min distance between center axis and four third lines
        # calculate the score according to 2 distance

        # use a parameter to enlarge result
        score = np.exp(- min_distance)

        return score
