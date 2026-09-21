# coding: utf-8

import numpy as np
import collections
from common.layers import EmbeddingDot, SigmoidWithLoss

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
# 这里就涉及到forward/backward的区别
class NegativeSamplingLoss():
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # 注意看layers的设计
        self.loss_layers = [SigmoidWithLoss()  for _ in range(self.sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(self.sample_size + 1)  ]

        self.params = []
        self.grads = []

        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_samples(target)

        # 下面分别是正负例的代码
        # 其实正常来说 如果是在线infer 那么其实没有sample的概念
        # 或者说这里的sample 不是 我已经推荐的sample
        # 就是feature set 然后下面是 label
        # infer没有必要推理负例 没错 但是word2vec主要是为了获取emb
        # infer真正的作用不是为了推理
        # infer真正的作用而是为了给backward缓存一些变量
        # 所有 如果是真正的在线服务 可以写两个infer 一个是在线推理 一个是backward时先forward缓存

        # positive sample
        correct_label = np.ones(batch_size, dtype=np.int32)
        score = self.embed_dot_layers[0].forward(h, target)
        loss = self.loss_layers[0].forward(score, correct_label)

        # negative sample
        # + - -
        # + - -
        # + - -
        # + - -
        # + - -
        # batch size = 5 sample size = 2
        # 下面的代码 按照sample size遍历 其实就是按例遍历
        # 第一列 拿出来5个sample的第一个负例 给第一层
        # 是这么个意思 批处理计算了
        # 这样其实也就不care batch了
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        # 上面的计算 完全按照定义算的
        # 先算score 正常推理 不管是正例还是负例
        # 然后算loss = score - label 计算残差
        # score是经过sigmoid的结果 label就是0 and 1 自己模拟给出即可
        # target是idx 取target emb跟上下文emb h一起算

        return loss

    def backward(self, dout = 1):
        dh = 0
        for emb_dot_layer, loss_layer in zip(self.embed_dot_layers, self.loss_layers):
            dscore = loss_layer.backward(dout)
            dh += emb_dot_layer.backward(dscore)

        # 写这种相对复杂的模型代码
        # 你就发现 内部嵌套层 每一个的forward/backward输入输出
        # 一定要搞清楚 才容易理解
        return dh