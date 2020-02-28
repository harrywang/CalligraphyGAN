from bert_serving.client import BertClient
import time
import numpy as np


class BertClientQuery:
    def __init__(self, words, topk=10, ip='localhost'):
        self.bc = BertClient(ip=ip)
        self.doc_vecs = self.bc.encode(words)
        self.topk = topk

    def query(self, string, topk=10):
        query = str(int(time.time())) + string
        query_vec = self.bc.encode([query])[0]
        # compute normalized dot product as score
        score = np.sum(query_vec * self.doc_vecs, axis=1) / np.linalg.norm(self.doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]

        return topk_idx
