# coding: utf-8
import sys

from ch8.attention_layer import TimeAttention

sys.path.append('..')
import numpy as np

from ch7.seq2seq import Encoder, Seq2seq
from common.time_layers import *
from attention_layer import TimeAttention

class AttentionEncoder(Encoder):
    def forward(self, xs):
        # 举例: N = 2, T = 3, H = 4
        # hs[0] = [[h_0], [h_1], [h_2]]     ← 样本
        # hs[1] = [[h_0], [h_1], [h_2]]     ← 样本
        # 所有样本 取最后一个时间步
        # 每一个序列 取最后一个输出
        # 这个是传统的encoder
        # attention encoder反而更简单

        xs = self.embed_layer.forward(xs)
        hs = self.lstm_layer.forward(xs)
        return hs

    # forward的输出看似是hs
    # 其实不是 最后输出的只是每个序列的最后一个输出
    def backward(self, dhs):
        dout = self.lstm_layer.backward(dhs)
        dout = self.embed_layer.backward(dout)
        return dout

class AttentionDecoder():
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
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        # Affine layer权重 这里没有weight typing
        # Attention decoder这里 affine layer有两个输入
        # 其一是encoder的通用上下文 经过attention计算后的 h
        # 其二是decoder目标序列的  序列编码h
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed_layer = TimeEmbedding(embed_W)
        self.lstm_layer = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention_layer = TimeAttention()
        self.affine_layer = TimeAffine(affine_W, affine_b)

        # 整理参数
        layers = [self.embed_layer, self.lstm_layer, self.attention_layer, self.affine_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    # 这个函数还是为了训练
    # 所以可以拿到xs decoder的序列都拿到了
    def forward(self, xs, enc_hs):
        h = enc_hs[:,-1]
        self.lstm_layer.set_state(h)

        out = self.embed_layer.forward(xs)
        dec_hs = self.lstm_layer.forward(out)

        # 这个c也是所有时间步的序列
        c = self.attention_layer.forward(enc_hs, dec_hs)

        # 每个时间步 把c的对应向量 和 decoder输出的hs 评起来
        # 这些都是time layers 一次计算t步
        out = np.concatenate((c, dec_hs), axis=2)

        # 这个score我觉得也没啥用 推理的时候肯定是不用的
        # 这里forward也是为了算一些cache
        # backward时候用
        score = self.affine_layer.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine_layer.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:,:,:H], dout[:,:,H:]
        denc_hs, ddec_hs1 = self.attention_layer.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm_layer.backward(ddec_hs)
        dh = self.lstm_layer.dh
        denc_hs[:, -1] += dh
        self.embed_layer.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id

        h = enc_hs[:,-1]
        self.lstm_layer.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            # 这里x的输入 虽然看起里是xs
            # 但其实就一个 这个是单步
            # 所以不能直接调用forward
            out = self.embed_layer.forward(x)
            decoder_hs = self.lstm_layer.forward(out)
            c = self.attention_layer.forward(enc_hs, decoder_hs)
            out = np.concatenate((c, decoder_hs), axis=2)
            score = self.affine_layer.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled

class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        self.encoder = AttentionEncoder(vocab_size, wordvec_size, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, wordvec_size, hidden_size)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads