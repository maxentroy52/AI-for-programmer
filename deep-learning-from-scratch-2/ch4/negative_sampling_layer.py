# coding: utf-8

import numpy as np
import collections
from common.layers import EmbeddingDot

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # 去重 计算vocab_size
        counter = collections.Counter()
        for word_id in corpus:
            counter[word_id] += 1
        self.vocab_size = len(counter)

        # 按照词频计算概率 后续按照词频采样做准备
        self.word_p = np.zeros(self.vocab_size)
        for i in range(self.vocab_size):
            self.word_p[i] = counter[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_samples(self, target):
        batch_size = target.shape[0]
        negative_samples = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            # 这里其实是一个重要操作
            # 正例的概率设置为0 然后只采样负例
            p[target_idx] = 0
            p /= p.sum()

            negative_samples[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_samples

# 这个层 有点东西
# 注意看书上的网络图解
# 负采样的个数 直接影响网络结构
# 我注释写的很清楚 通过 正例 + 部分负例(采样) 模拟整个词表的概率分布
# 注意 不是像softmax那样获取整个词表的概率分布
# 而是 单一词的概率分布 不至于对除了target word之外的词 一无所知
# 本质学出来w_out不一致
class NegativeSamplingLoss():