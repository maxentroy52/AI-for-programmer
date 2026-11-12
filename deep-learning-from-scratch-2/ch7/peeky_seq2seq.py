# coding: utf-8

import numpy as np
from seq2seq import Encoder, Seq2seq
from common.time_layers import *
from common.base_model import BaseModel

class PeekyDecoder(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 初始化权重
        # TimeEmbedding layer权重
        embed_W = (rn(V, D) / 100).astype('f')

        # LSTM layer权重
        # 矩阵合并 便于计算(不必频繁访存取操作数)
        # LSTM改进了RNN 遗忘门 新信息 输入门 输出门
        # 4个门的计算 合并为一个大矩阵计算
        lstm_Wx = (rn(D + H, 4 * H) / np.sqrt(D + H)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        # Affine layer权重 这里没有weight typing
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed_layer = TimeEmbedding(embed_W)
        self.lstm_layer = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine_layer = TimeAffine(affine_W, affine_b)

        # 整理参数
        self.params, self.grads = [], []
        for layer in (self.embed_layer, self.lstm_layer, self.affine_layer):
            self.params += layer.params
            self.grads += layer.grads

        self.cache = None

    # 这里注意 forward本质给training使用
    # 所以这里处理的还是time layers得数据 所以处理序列
    # 单独的生成交给generate.
    def forward(self, xs, h):
        self.lstm_layer.set_state(h)

        N, T = xs.shape
        N, H = h.shape
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)

        out = self.embed_layer.forward(xs)

        # 通用context给到lstm的每一个时间步
        out = np.concatenate((hs, out), axis=2)
        out = self.lstm_layer.forward(out)

        # 通用context给到affine的每一个时间步
        out = np.concatenate((hs, out), axis=2)
        score = self.affine_layer.forward(out)

        self.cache = H

        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine_layer.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm_layer.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed_layer.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm_layer.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        """
        xs = [
            [10, 25, 3, 88, 7],      # 样本 0 的 5 个时间步
            [42, 5, 19, 60, 33],     # 样本 1 的 5 个时间步
        ]

        xs.shape = (2, 5)
                    ↑  ↑
                    N  T
        x=[3]

        x = [
            [3]
        ]
        一个序列，序列长度为1.里面的token id是3
        """
        self.lstm_layer.set_state(h)
        sampled = []
        sample_id = start_id

        # generate时候 单步
        # 原本(N,H)是(1,H)
        # 这里再加一个时间步 本质是(N,T,H)
        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed_layer.forward(x)

            # 通用context给到lstm的每一个时间步
            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm_layer.forward(out)

            # 通用context给到affine的每一个时间步
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine_layer.forward(out)

            # 这里自回归
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)

        # decoder里面本来应该包含这一部分
        # 但是 decoder实现了forward跟backward共有的部分
        # 所以 这一块放到seq2seq来实现
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads