"""
This file implemented Aesthetic Filters.

AestheticFilter is the base class and it has 3 method, _get_score, _score_2_index and get_result.

Function get_result implemented in AestheticFilter should not be override.
In `get_result`, we firstly use _get_score to calculate scores of every image, and then use _score_2_index to sort them
according to the need. Finally return the top-k result. This is general idea of Aesthetic Filters.

But how to get scores and sort them are different according to different purpose. So, subclasses can do different things
by override `_ger_score` and `_score_2_index`.

ThresholdFilter handle a special condition in which we want to filter results with threshold, which means we just want
to get results between upper bound and lower bound.
In this implementation, I use a value's minimum distance to boundaries to sort them. The distance is calculated as
follows:
    d = min( value - lower-bound, upper-bound - value). So we just need to return positive distance.

WhiteSpaceFilter is a special case of ThresholdFilter.
"""
import numpy as np


class AestheticFilter(object):
    def __init__(self):
        pass

    def _get_score(self, input_image):
        """
        :param input_image:
        :return: a list of scores
        """
        return range(len(input_image))

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
        scores = self._get_score(input_image)
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

    def _get_score(self, input_image):
        scores = []
        for img in input_image:
            width, height = img.shape[0], img.shape[1]
            white_space_cnt = 0.
            for i in range(width):
                for j in range(height):
                    if img[i][j] >= self.white_threshold:
                        white_space_cnt += 1
            scores.append(white_space_cnt / (width * height))

        return scores
