import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import numpy as np
import time
from models.calligraphyGAN_dataset import _get_embedding


class BertQuery:
    def __init__(self, model_dir, bq_result_path=None):
        """

        :param model_dir: if no cached model in this dir, transformers will download them automatically
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = TFBertModel.from_pretrained('bert-base-chinese', cache_dir=model_dir)

        if bq_result_path is None:
            self.doc_vecs = []

            words = _get_embedding('./data/label_character.csv')
            # _get_embedding return the dict like { ..., character: embedding, ... }
            # convert to a list like [ ..., character, ... ]
            words = list(words.keys())

            for i, word in enumerate(words):
                print('%d / %d' % (i, len(words)))
                temp_output = self.model(tf.constant(self.tokenizer.encode(word, add_special_tokens=True))[None, :])
                self.doc_vecs.append(temp_output[1])
            self.doc_vecs = np.array(self.doc_vecs)
            self.doc_vecs = np.squeeze(self.doc_vecs, axis=(1,))
            np.save('./data/bq-result.npy', self.doc_vecs)
        else:
            self.doc_vecs = np.load('./data/bq-result.npy')

    def query(self, text, topk=10):
        query = str(time.time()) + text
        query_vec = self.model(tf.constant(self.tokenizer.encode(query, add_special_tokens=True))[None, :])[1]

        # compute normalized dot product as score
        score = np.sum(query_vec * self.doc_vecs, axis=1) / np.linalg.norm(self.doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]

        return topk_idx
