import sys, os
import numpy as np

sys.path.append(os.pardir)
from common import *

# step1 - mini-batch
# step2 - calculating gradients
# step3 - update parameters(gradients descent)
# step4 - repeating

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}

        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        # net.loss总共有3个参数x, t, w.
        # 对于每一个样本，(x,t)都是确定的。
        # 此时，当前的w可以用x算出y，这是推理的过程。
        # training就是，把x,t看做常数。w是变量，当前的w是一个position
        # 然后算出当前的一个梯度grad(w)
        #
        # 具体计算的时候两种办法
        # 1.可以数值微分的办法(给当前w一个h, 然后计算)
        # 2.可以解析法(直接计算出导函数，带入当前w)
        #
        # 这个地方不好理解的是多元函数，到底谁是自变量，谁是因变量
        #
        # 这个损失函数你也好理解
        # 它是可以带入一个，也可以带入所有
        # 带入所有的(x, t)之后，形成关于w的函数
        loss_w = lambda w: self.loss(x, t)

        grads = []