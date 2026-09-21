import numpy as np

from common.layers import Embedding
from ch4.negative_sampling_layer import NegativeSamplingLoss

class CBOW():
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 模型初始化
        # 这可是模型啊 初始化 纯random 是不是挺扯得
        # w_in and w_out有一样的结构 之前说过了 因为这里引入emb层 直接根据idx取
        W_in = np.random.randn(V, H).astype('f')
        W_out = np.random.randn(V, H).astype('f')

        # 生成层
        # ns_loss并不是纯粹的Loss layer
        # 因为它包了一层embedding dot计算score 这个其实是forward的工作
        # 传统的forward 其实就是计算score
        # training时才需要softmax with loss 这里是sigmoid with loss
        # 但是ns_loss把score计算也包括进去了
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power = 0.75, sample_size = 5)

        # 统一管理
        # 本质还是为了统一管理params and grads
        self.layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, targets):
        # contexts targets
        # [you, goodbye] [say]   [ [0,2]   [1,      [ [[1,0,0,0,0,0],[0,0,1,0,0,0]],     [[0,1,0,0,0,0],
        # [say, and] [goodbye]     [1,3],   2,
        # [goodbye, i] [and]       [2,4],   3,
        # [and, say] [i]           [3,5]]   4]        [[0,0,0,1,0,0],[0,0,,0,0,1]]]      [[0,0,0,1,0,0]]
        #
        # 这里其实one hot就不用了 因为one hot 和 w_in的点乘结果就是把第idx行取出来
        # 所以 这里直接取了 底层Emb替代MatMul
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        h *= 1 / len(self.in_layers)

        # 严格意义上来说 这里的infer不是给在线用的
        # 纯backward计算loss用
        loss = self.ns_loss.forward(h, targets)
        return loss

    def backward(self, dout = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None