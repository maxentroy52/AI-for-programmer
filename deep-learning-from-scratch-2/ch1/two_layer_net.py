import sys
sys.path.append("..")

import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        ## part1
        ## 模型总共3部分 (超参-参数-网络结构)

        ## 超参
        ## 对于分类问题 input size是特征数 output size是分类数 这两个其实是固定的 因为数据决定
        ## hidden size中间神经元个数 这是超参
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 初始化权重和偏置
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        # 网络搭建 - 生成层
        # 拆开的原因是infer只需要affine-sigmoid-affine
        # 这本质是个多分类问题，看谁分大就行
        # 只有在train的时候，才需要loss(cross entropy) with softmax
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        # 所以, loss function其实是模型的一部分
        self.loss_layer = SoftmaxWithLoss()

        ## part2
        ## 为了方便代码操作
        ## 将参数，梯度都汇总到一块
        ## 主要是便于计算
        ## 下面其实是基操
        ## 因为在layer的抽象当中，params/grads都是核心模型参数
        ## 下面的代码，是列表拼接，自然拼到了一起
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # forward impl的实现
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    # 这个输出其实没啥用
    # 本质其实是把W1 b1 W2 b2的梯度算出来
    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout