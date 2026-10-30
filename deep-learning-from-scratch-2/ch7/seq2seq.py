# coding: utf-8
import sys
sys.path.append('..')

from common.time_layers import *
from common.base_model import BaseModel

class Encoder(BaseModel):
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

        # 生成层
        self.embed_layer = TimeEmbedding(embed_W)
        self.lstm_layer = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        # 参数/梯度
        self.params = self.embed_layer.params + self.lstm_layer.params
        self.grads = self.embed_layer.grads + self.lstm_layer.grads
        self.hs = None

    def forward(self,xs):
        # 举例: N = 2, T = 3, H = 4
        # hs[0] = [[h_0], [h_1], [h_2]]     ← 样本
        # hs[1] = [[h_0], [h_1], [h_2]]     ← 样本
        # 所有样本 取最后一个时间步
        # 每一个序列 取最后一个输出

        xs = self.embed_layer.forward(xs)
        hs = self.lstm_layer.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    # forward的输出看似是hs
    # 其实不是 最后输出的只是每个序列的最后一个输出
    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:,-1,:] = dh

        dout = self.lstm_layer.backward(dhs)
        dout = self.embed_layer.backward(dout)

        return dout

class Decoder(BaseModel):
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
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed_layer = TimeEmbedding(embed_W)
        self.lstm_layer = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine_layer = TimeAffine(affine_W, affine_b)

        # 整理参数
        self.params, self.grads = [], []
        for layer in (self.embed_layer, self.lstm_layer, self.affine_layer):
            self.params += layer.params
            self.grads += layer.grads

    # 这里注意 forward本质给training使用
    # 所以这里处理的还是time layers得数据 所以处理序列
    # 单独的生成交给generate.
    def forward(self, xs, h):
        self.lstm_layer.set_state(h)

        out = self.embed_layer.forward(xs)
        out = self.lstm_layer.forward(out)
        score = self.affine_layer.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine_layer.backward(dscore)
        dout = self.lstm_layer.backward(dout)
        dout = self.embed_layer.backward(dout)

        dh = self.lstm_layer.dh
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
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed_layer.forward(x)
            out = self.lstm_layer.forward(out)
            score = self.affine_layer.forward(out)

            # 这里自回归
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled